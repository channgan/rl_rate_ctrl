"""Own one PX4 SITL + Gazebo server per episode, using localhost synchronous IPC."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np
from bridge_transport import choose_bridge_port, bridge_ready


class SimulatorError(RuntimeError):
    """Infrastructure failure; never turn this into a valid training transition."""


class GazeboPX4Backend:
    def __init__(self, config_path: str | Path, log_dir: str | Path | None = None, *, instance: int = 41):
        if type(instance) is not int or not 0 <= instance <= 100:
            raise ValueError("Invalid PX4 instance")
        self.instance = instance
        self.config = json.loads(Path(config_path).read_text())
        self.log_dir = Path(log_dir or Path(self.config["runtime"]) / "episodes").resolve()
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.processes: list[subprocess.Popen] = []
        self.files = []
        self.sock = self.stream = None
        self.sequence = 0
        self.last_sim_us = 0
        self.episode_dir: Path | None = None
        self.instance_lock = None
        self.native_outer = False
        self.native_baseline = False
        self.native_torque = False

    @property
    def hover(self):
        return float(self.config["hover_speed_fraction"])

    @property
    def px4_hover_command(self):
        # x500 native ESC range is 150..1000 rad/s with THR_MDL_FAC=0.
        # This is a calibrated PX4 collective command, not F/Fmax or PWM.
        minimum = 150. / float(self.config['max_rotor_rad_s'])
        return (self.hover - minimum) / (1. - minimum)

    def set_mode(self, mode):
        """Change assets between episodes; reset() owns process replacement."""
        if mode not in ("ball_rig", "free_flight"):
            raise ValueError("Unknown simulation mode")
        if mode == self.config["mode"]:
            return
        path = Path(self.config["runtime"]) / f"runtime.{mode}.json"
        config = json.loads(path.read_text())
        for field in ("hover_speed_fraction", "mass_kg", "max_rotor_rad_s", "control_dt_s", "px4_commit",
                      "thrust_coefficient", "gravity_m_s2", "snapshot_version"):
            if config[field] != self.config[field]:
                raise ValueError(f"Stage asset mismatch: {field}")
        if config["mode"] != mode:
            raise ValueError("Stage configuration has wrong mode")
        self.close()
        self.config = config

    def _spawn(self, args, env, name):
        output = (self.episode_dir / f"{name}.log").open("w")
        self.files.append(output)
        self.processes.append(subprocess.Popen(args, cwd=self.episode_dir, env=env,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            start_new_session=True))

    def reset(self, seed: int | None = None):
        # Bounded retries only before an initial observation is returned.
        # No action transition or reward exists yet; preserve the episode seed.
        for attempt in range(3):
            try:
                return self._reset_once(seed)
            except SimulatorError as exc:
                log = self.episode_dir / "px4.log" if self.episode_dir else None
                bind_failed = log is not None and log.is_file() and "RL bridge bind failed" in log.read_text(errors="replace")
                incomplete_init = str(exc) == "Incomplete simulator response"
                native_init_failed = str(exc).startswith("PX4 native outer initialization failed; logs:")
                # PX4/gz transport may segfault while constructing a new episode.
                # Retry only inside reset, before any policy observation is issued;
                # a crash during step still aborts instead of inventing a sample.
                startup_segfault = str(exc).startswith("Simulator process exited (-11);")
                if not (bind_failed or incomplete_init or native_init_failed or startup_segfault) or attempt == 2:
                    raise
                print(f"Bridge initialization failed ({exc}); retry {attempt + 1}/2 with a new port; logs: {self.episode_dir}", flush=True)

    def _reset_once(self, seed: int | None = None):
        self.close()
        try:
            import fcntl
            self.instance_lock = open(f"/tmp/px4_rl_instance{self.instance}.lock", "w")
            try:
                fcntl.flock(self.instance_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise SimulatorError(f"Another training environment owns PX4 instance {self.instance}") from exc
            token = uuid.uuid4().hex[:12]
            self.episode_dir = self.log_dir / token
            self.episode_dir.mkdir()
            startup = self.episode_dir / "startup.sh"
            startup.write_text("#!/bin/sh\n. etc/init.d-posix/rcS\necho ready > rl_startup.ready\n")
            px4 = Path(self.config["px4"])
            build = px4 / "build/px4_sitl_default"
            binary = build / "bin/px4"
            if not binary.is_file():
                raise SimulatorError(f"PX4 binary missing: {binary}; run setup first")
            port = choose_bridge_port()
            env = os.environ.copy()
            env.update(GZ_PARTITION=f"rate_rl_{token}", GZ_IP="127.0.0.1", HEADLESS="1",
                       RATE_RL_NOISE_SEED=str(seed or 0),
                       PX4_RL_PORT=str(port), PX4_GZ_STANDALONE="1", PX4_GZ_WORLD="rate_training",
                       PX4_GZ_MODEL_NAME="x500_rl_0", PX4_SYS_AUTOSTART="4001", PX4_SIM_MODEL="gz_x500",
                       PX4_PARAM_IMU_GYRO_RATEMAX="1000", PX4_PARAM_IMU_INTEG_RATE="1000",
                       PX4_PARAM_SIM_BAT_DRAIN="0", PX4_PARAM_UXRCE_DDS_CFG="0")
            env["GZ_SIM_RESOURCE_PATH"] = ":".join([self.config["models"], str(px4 / "Tools/simulation/gz/models")])
            if self.native_outer:
                env.update(PX4_RL_NATIVE_OUTER="1", PX4_PARAM_MPC_THR_HOVER=str(self.px4_hover_command),
                           PX4_PARAM_MPC_USE_HTE="0", PX4_PARAM_MC_ROLLRATE_MAX="57.29578",
                           PX4_PARAM_MC_PITCHRATE_MAX="57.29578", PX4_PARAM_MC_YAWRATE_MAX="57.29578",
                           PX4_PARAM_COM_RC_IN_MODE="4", PX4_PARAM_NAV_DLL_ACT="0",
                           PX4_PARAM_IMU_GYRO_CUTOFF="80", PX4_PARAM_IMU_DGYRO_CUTOFF="40",
                           PX4_PARAM_SIM_BAT_DRAIN="86400", PX4_PARAM_THR_MDL_FAC="0")
                if self.native_baseline:
                    env['PX4_RL_NATIVE_BASELINE'] = '1'
                if self.native_torque:
                    if self.native_baseline:
                        raise ValueError('Torque policy and native rate baseline are mutually exclusive')
                    env['PX4_RL_NATIVE_TORQUE'] = '1'
                # Airframe set-default commands can restore zero-valued settings.
                # Apply native training settings after rcS, before accepting INIT.
                keys = ['MPC_THR_HOVER','MPC_USE_HTE','MC_ROLLRATE_MAX','MC_PITCHRATE_MAX',
                        'MC_YAWRATE_MAX','COM_RC_IN_MODE','NAV_DLL_ACT','SIM_BAT_DRAIN','THR_MDL_FAC',
                        'IMU_GYRO_CUTOFF','IMU_DGYRO_CUTOFF']
                startup.write_text('#!/bin/sh\n. etc/init.d-posix/rcS\n' +
                                   ''.join(f'param set {key} {env["PX4_PARAM_"+key]}\n' for key in keys) +
                                   'echo ready > rl_startup.ready\n')
            env["GZ_SIM_SYSTEM_PLUGIN_PATH"] = ":".join([self.config["plugin_dir"], str(build / "src/modules/simulation/gz_plugins")])
            # Each instance owns its lock and PX4 ports; partition isolates Gazebo.
            world_path = self.config["world"]
            if self.native_outer:
                world = ET.parse(world_path)
                vehicle = next(x for x in world.findall("world/include") if x.findtext("name") == "x500_rl_0")
                vehicle.find("pose").text = "0 0 0 0 0 0"
                world_path = str(self.episode_dir / "takeoff_world.sdf")
                world.write(world_path, encoding="unicode")
            self._spawn(["gz", "sim", "-r", "-s", "--seed", str(seed or 0), world_path], env, "gazebo")
            self._spawn([str(binary), "-d", "-i", str(self.instance), "-s", str(startup),
                         "-w", str(self.episode_dir), str(build / "etc")], env, "px4")
            deadline = time.monotonic() + 60
            while True:
                self._check_processes()
                try:
                    ready = bridge_ready(self.episode_dir / "px4.log")
                except RuntimeError as exc:
                    raise SimulatorError(str(exc)) from exc
                if not ready:
                    if time.monotonic() > deadline:
                        raise SimulatorError(f"PX4 bridge readiness timed out; inspect {self.episode_dir}")
                    time.sleep(0.05)
                    continue
                try:
                    self.sock = socket.create_connection(("127.0.0.1", port), timeout=1)
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise SimulatorError(f"PX4 bridge startup timed out; inspect {self.episode_dir}")
                    time.sleep(0.1)
            while not (self.episode_dir / "rl_startup.ready").is_file():
                self._check_processes()
                if time.monotonic() > deadline:
                    raise SimulatorError(f"PX4 startup script timed out; inspect {self.episode_dir}")
                time.sleep(0.05)
            self.sock.settimeout(130 if self.native_outer else 20)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.stream = self.sock.makefile("rwb", buffering=0)
            self.sequence = 0
            initial = self._request("INIT")
            if initial["seq"] != 0 or initial["action_seq"] != 0:
                raise SimulatorError("Initial action sequence must be zero")
            if not self.native_outer and not np.allclose(initial["applied_pwm"], self.hover, rtol=0, atol=1e-6):
                raise SimulatorError("Initial PWM does not match bootstrap hover command")
            if not self.native_outer and not np.allclose(initial["rotor_speed_fraction"], initial["applied_pwm"], rtol=0, atol=0.005):
                raise SimulatorError("Initial rotor dynamics have not settled")
            self.last_sim_us = initial["sim_us"]
            return initial
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                # Initialization remains the primary error, while a cleanup
                # failure is retained as its cause instead of hiding the fault.
                raise error from cleanup_error
            raise

    def _check_processes(self):
        for process in self.processes:
            if process.poll() is not None:
                raise SimulatorError(f"Simulator process exited ({process.returncode}); inspect {self.episode_dir}")

    def _request(self, line):
        self._check_processes()
        if self.stream is None:
            raise SimulatorError("Call reset before stepping")
        try:
            self.stream.write((line + "\n").encode("ascii"))
            raw = self.stream.readline(8192)
            if not raw.endswith(b"\n"):
                raise SimulatorError("Incomplete simulator response")
            data = json.loads(raw)
            if "error" in data:
                raise SimulatorError(f"{data['error']}; logs: {self.episode_dir}")
            for name, size in [("rates", 3), ("rates_true", 3), ("position_ned", 3),
                               ("linear_velocity_ned", 3), ("q_ned_frd", 4),
                               ("rotor_speed_fraction", 4), ("applied_pwm", 4), ("gyro_noise_flu", 10)]:
                values = np.asarray(data[name], dtype=np.float64)
                if values.shape != (size,) or not np.all(np.isfinite(values)):
                    raise SimulatorError(f"Invalid {name} feedback")
            if not np.isclose(np.linalg.norm(data["q_ned_frd"]), 1.0, atol=1e-5):
                raise SimulatorError("Invalid attitude quaternion")
            if data["snapshot_version"] != 2 or data["snapshot_valid"] is not True:
                raise SimulatorError("Incomplete or incompatible truth snapshot")
            for name in ("seq", "action_seq", "sim_us", "sample_us", "source_sample_us", "truth_us"):
                if type(data[name]) is not int or data[name] < 0:
                    raise SimulatorError(f"Invalid {name}")
            if data["sim_us"] == 0 or any(data[name] != data["sim_us"] for name in
                    ("sample_us", "source_sample_us", "truth_us")):
                raise SimulatorError("Gyro source/truth/simulation time mismatch")
            if data["gyro_noise_flu"][0] != data["sim_us"]:
                raise SimulatorError("Gazebo gyro noise sample time mismatch")
            if data["action_seq"] != data["seq"]:
                raise SimulatorError("Truth snapshot belongs to a different action")
            if np.any(np.asarray(data["applied_pwm"]) < 0) or np.any(np.asarray(data["applied_pwm"]) > 1):
                raise SimulatorError("Invalid applied PWM feedback")
            if np.any(np.asarray(data["rotor_speed_fraction"]) < 0):
                raise SimulatorError("Invalid rotor speed feedback")
            if self.native_outer:
                for name, size in [('px4_rate_target',3), ('px4_local_position',3), ('px4_attitude_target',4)]:
                    v=np.asarray(data[name],dtype=float)
                    if v.shape!=(size,) or not np.isfinite(v).all():
                        raise SimulatorError(f'Invalid native PX4 feedback: {name}')
                age=data['px4_now_us']-data['px4_reference_us']
                if (data['px4_position_control'] and not 0 <= age <= 50000) or not np.isfinite(data['px4_thrust_body_z']):
                    raise SimulatorError('Native PX4 reference stale or invalid')
                if self.native_torque:
                    if not data.get('px4_policy_owns_torque') or not data.get('px4_native_allocator'):
                        raise SimulatorError('RL torque ownership / native allocation not confirmed')
                    torque = np.asarray(data.get('applied_torque'), dtype=float)
                    allocated = np.asarray(data.get('allocated_pwm'), dtype=float)
                    if torque.shape != (3,) or not np.isfinite(torque).all() or np.any(np.abs(torque) > 1.):
                        raise SimulatorError('Invalid applied torque feedback')
                    if allocated.shape != (4,) or not np.isfinite(allocated).all() or np.any((allocated < 0.) | (allocated > 1.)):
                        raise SimulatorError('Invalid native allocated PWM feedback')
                    data['applied_speed_fraction'] = data['applied_pwm']
                    data['applied_pwm'] = allocated.tolist()
                elif not self.native_baseline and not data['px4_policy_owns_motors']:
                    raise SimulatorError('Native rate controller/allocator were not stopped')
            return data
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise SimulatorError(f"Invalid bridge exchange: {exc}; logs: {self.episode_dir}") from exc

    def step(self, motors: np.ndarray, dt: float):
        if self.native_torque:
            raise SimulatorError('Direct motor actions are disabled in native torque mode')
        motors = np.asarray(motors, dtype=np.float64)
        if motors.shape != (4,) or not np.all(np.isfinite(motors)) or np.any(motors < 0) or np.any(motors > 1):
            raise ValueError("Motor action must contain four finite PWM fractions in [0, 1]")
        steps = round(dt / 0.001)
        if not 1 <= steps <= 100 or not np.isclose(steps * 0.001, dt):
            raise ValueError("dt must be an integer multiple of 1 ms, up to 100 ms")
        seq = self.sequence + 1
        result = self._request(f"STEP {seq} {steps} " + " ".join(f"{v:.9g}" for v in motors))
        if result["seq"] != seq or result["sim_us"] != self.last_sim_us + steps * 1000:
            raise SimulatorError("Sequence or simulation time mismatch")
        if not np.allclose(result["applied_pwm"], motors, rtol=0, atol=1e-6):
            raise SimulatorError("Simulator applied a different PWM action")
        self.sequence = seq
        self.last_sim_us = result["sim_us"]
        return result

    def snapshot(self):
        result = self._request("STATE")
        if result["seq"] != self.sequence or result["sim_us"] != self.last_sim_us:
            raise SimulatorError("Paused snapshot changed sequence or simulation time")
        return result

    def step_torque(self, torque: np.ndarray, dt: float):
        """Request FRD normalized torque; PX4 supplies thrust and allocation."""
        if not self.native_outer or not self.native_torque or self.native_baseline:
            raise SimulatorError('Native torque control mode is not enabled')
        torque = np.asarray(torque, dtype=np.float64)
        if torque.shape != (3,) or not np.all(np.isfinite(torque)) or np.any(np.abs(torque) > 1.):
            raise ValueError('Torque action must contain three finite normalized values in [-1,1]')
        steps = round(dt / .001)
        if not 1 <= steps <= 100 or not np.isclose(steps * .001, dt):
            raise ValueError('dt must be an integer multiple of 1 ms, up to 100 ms')
        seq = self.sequence + 1
        result = self._request(f'STEP_TORQUE {seq} {steps} ' + ' '.join(f'{v:.9g}' for v in torque))
        if result['seq'] != seq or result['sim_us'] != self.last_sim_us + steps * 1000:
            raise SimulatorError('Torque sequence or simulation time mismatch')
        if not np.allclose(result['applied_torque'], torque, rtol=0, atol=1e-7):
            raise SimulatorError('PX4 applied a different torque action')
        if (result.get('allocator_action_matched') is not True
                or result.get('allocator_first_sample_us') != self.last_sim_us
                or result.get('allocator_action_sample_us') != self.last_sim_us
                or not self.last_sim_us <= result.get('allocator_last_sample_us', -1) <= result['sim_us']):
            raise SimulatorError('Native allocator did not acknowledge this torque before physics advance')
        forwarded = np.asarray(result.get('forwarded_thrust_body'), dtype=float)
        if (forwarded.shape != (3,) or not np.isfinite(forwarded).all()
                or result.get('forwarded_thrust_sample_us') != result['allocator_last_sample_us']):
            raise SimulatorError('Native thrust allocation feedback missing or mismatched')
        self.sequence = seq
        self.last_sim_us = result['sim_us']
        return result

    def set_position_goal(self, position_ned, heading):
        if not self.native_outer:
            raise SimulatorError('Native PX4 outer loop is not enabled')
        values = np.r_[position_ned, heading]
        if values.shape != (4,) or not np.isfinite(values).all():
            raise ValueError('Invalid position/heading goal')
        return self._request('GOAL ' + ' '.join(f'{v:.9g}' for v in values))

    def close(self):
        errors = []

        def close_resource(resource):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    errors.append(error)

        def signal_process(process, signum):
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass
            except BaseException as error:
                errors.append(error)

        stream, self.stream = self.stream, None
        sock, self.sock = self.sock, None
        close_resource(stream)
        close_resource(sock)
        # Only kill process groups created by this backend, never global px4/gz names.
        for process in reversed(self.processes):
            signal_process(process, signal.SIGTERM)
        for process in self.processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                signal_process(process, signal.SIGKILL)
                try:
                    process.wait(timeout=5)
                except BaseException as error:
                    errors.append(error)
            except BaseException as error:
                errors.append(error)
        self.processes.clear()
        for file in self.files:
            close_resource(file)
        self.files.clear()
        instance_lock, self.instance_lock = self.instance_lock, None
        close_resource(instance_lock)
        if errors:
            # Report failures only after every owned process/resource was given
            # a cleanup attempt, including the instance lock needed by workers.
            raise errors[0]
