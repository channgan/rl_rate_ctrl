"""Apply the opt-in SITL bridge and generate training assets outside PX4 source."""
import argparse
import json
import math
import pathlib
import shutil
import subprocess
import xml.etree.ElementTree as ET

PROJECT = pathlib.Path(__file__).resolve().parents[1]


def patch_native_torque_metadata(px4):
    """Carry actual allocator input through the native ESC output for lockstep ACKs.

    No allocation, motor mapping, thrust curve or output limits are replaced.
    The accessors expose the exact FunctionMotors sample consumed by MixingOutput.
    """
    def insert(path, marker, anchor, replacement):
        text = path.read_text()
        if marker in text:
            return
        if text.count(anchor) != 1:
            raise RuntimeError(f"Cannot apply native torque metadata patch: {path}")
        path.write_text(text.replace(anchor, replacement))

    lib = px4 / "src/lib/mixer_module"
    base = lib / "functions/FunctionProviderBase.hpp"
    anchor = "\tvirtual bool getLatestSampleTimestamp(hrt_abstime &t) const { return false; }"
    insert(base, "rlMotorSnapshot", anchor, anchor + "\n\n"
        "\tvirtual bool rlMotorSnapshot(hrt_abstime &t, float *controls, unsigned count) const { return false; }\n")
    motors = lib / "functions/FunctionMotors.hpp"
    insert(motors, "_rl_raw_data = _data", "\t\tif (_topic.update(&_data)) {",
        "\t\tif (_topic.update(&_data)) {\n\t\t\t_rl_raw_data = _data;")
    anchor = "\tbool getLatestSampleTimestamp(hrt_abstime &t) const override { t = _data.timestamp_sample; return t != 0; }"
    insert(motors, "bool rlMotorSnapshot", anchor, anchor + "\n\n"
        "\tbool rlMotorSnapshot(hrt_abstime &t, float *controls, unsigned count) const override\n"
        "\t{\n\t\tif (count > actuator_motors_s::NUM_CONTROLS) { return false; }\n"
        "\t\tt = _rl_raw_data.timestamp_sample;\n"
        "\t\tfor (unsigned i = 0; i < count; ++i) { controls[i] = _rl_raw_data.control[i]; }\n"
        "\t\treturn t != 0;\n\t}\n")
    insert(motors, "actuator_motors_s _rl_raw_data", "\tactuator_motors_s _data{};",
        "\tactuator_motors_s _data{};\n\tactuator_motors_s _rl_raw_data{};")
    mixer = lib / "mixer_module.hpp"
    anchor = "\tOutputFunction outputFunction(int index) const { return _function_assignment[index]; }"
    insert(mixer, "bool rlMotorSnapshot", anchor, anchor + "\n\n"
        "\tbool rlMotorSnapshot(hrt_abstime &t, float *controls, unsigned count) const\n"
        "\t{\n\t\treturn _function_allocated[0] && _function_allocated[0]->rlMotorSnapshot(t, controls, count);\n\t}\n")
    esc = px4 / "src/modules/simulation/gz_bridge/GZMixingInterfaceESC.cpp"
    anchor = "\t\tif (_actuators_pub.Valid()) {"
    insert(esc, "rl_allocated_controls", anchor,
        "\t\tif (std::getenv(\"PX4_RL_NATIVE_TORQUE\")) {\n"
        "\t\t\thrt_abstime sample = 0; float controls[4]{};\n"
        "\t\t\tif (_mixing_output.rlMotorSnapshot(sample, controls, 4)) {\n"
        "\t\t\t\tauto *header = rotor_velocity_message.mutable_header();\n"
        "\t\t\t\theader->mutable_stamp()->set_sec(sample / 1000000);\n"
        "\t\t\t\theader->mutable_stamp()->set_nsec((sample % 1000000) * 1000);\n"
        "\t\t\t\tauto *data = header->add_data(); data->set_key(\"rl_allocated_controls\");\n"
        "\t\t\t\tfor (float c : controls) { data->add_value(std::to_string(c)); }\n"
        "\t\t\t}\n\t\t}\n" + anchor)
    text = esc.read_text()
    if "#include <cstdlib>" not in text:
        esc.write_text("#include <cstdlib>\n" + text)


def prepare_mode(px4, runtime, mode):
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=px4, text=True).strip()
    expected = subprocess.check_output(["git", "rev-parse", "v1.17.0^{commit}"], cwd=px4, text=True).strip()
    if revision != expected:
        raise RuntimeError("This bridge is tested against PX4 v1.17.0; use an isolated checkout.")
    patch_native_torque_metadata(px4)
    directory = px4 / "src/modules/simulation/gz_bridge"
    shutil.copy2(PROJECT / "px4/RLBridge.cpp", directory / "RLBridge.cpp")
    cmake = directory / "CMakeLists.txt"
    text = cmake.read_text()
    if "RLBridge.cpp" not in text:
        text = text.replace("\t\t\tGZBridge.cpp", "\t\t\tGZBridge.cpp\n\t\t\tRLBridge.cpp")
        cmake.write_text(text)
    cpp = directory / "GZBridge.cpp"
    text = cpp.read_text()
    if "start_rl_bridge(_world_name)" not in text:
        text = text.replace('#include "GZBridge.hpp"', '#include "GZBridge.hpp"\nvoid start_rl_bridge(const std::string &world);')
        text = text.replace("\tScheduleNow();\n\treturn OK;", "\tstart_rl_bridge(_world_name);\n\tScheduleNow();\n\treturn OK;")
    imu_subscription = '\tif (!_node.Subscribe(imu_topic, &GZBridge::imuCallback, this))'
    if 'imu_topic += "/rl_noisy"' not in text:
        if imu_subscription not in text:
            raise RuntimeError("Cannot locate PX4 IMU subscription")
        text = text.replace(imu_subscription,
            '\tif (std::getenv("PX4_RL_PORT")) { imu_topic += "/rl_noisy"; }\n' + imu_subscription)
        if '#include <cstdlib>' not in text:
            text = '#include <cstdlib>\n' + text
    # Only this opt-in bridge uses the Gazebo IMU source time. Receipt time can
    # advance before a queued older sample is handled, which would make a stale
    # observation pass the synchronous-step check. Keep normal PX4 filtering.
    imu_start = text.index("void GZBridge::imuCallback(")
    imu_end = text.index("\nvoid GZBridge::", imu_start + 1)
    imu = text[imu_start:imu_end]
    if "const uint64_t timestamp_sample =" not in imu:
        declaration = "\tconst uint64_t timestamp = hrt_absolute_time();"
        if imu.count(declaration) != 1:
            raise RuntimeError("Cannot locate PX4 IMU sample timestamp")
        imu = imu.replace(declaration, declaration +
            '\n\tconst uint64_t timestamp_sample = std::getenv("PX4_RL_PORT")\n'
            '\t\t? uint64_t(msg.header().stamp().sec()) * 1000000 + msg.header().stamp().nsec() / 1000\n'
            '\t\t: timestamp;')
        if imu.count("timestamp_sample = timestamp;") != 2:
            raise RuntimeError("Unexpected PX4 IMU accel/gyro publication layout")
        imu = imu.replace("timestamp_sample = timestamp;", "timestamp_sample = timestamp_sample;")
        text = text[:imu_start] + imu + text[imu_end:]
    cpp.write_text(text)

    # Each mode owns its assets; generating free flight must not remove the
    # constraint from a rig model used by an existing training configuration.
    models = runtime / "assets" / mode / "models"
    worlds = runtime / "assets" / mode / "worlds"
    models.mkdir(parents=True, exist_ok=True)
    worlds.mkdir(parents=True, exist_ok=True)
    upstream = px4 / "Tools/simulation/gz/models"
    base_dir = models / "x500_rl_base"
    base_dir.mkdir(exist_ok=True)
    base = ET.parse(upstream / "x500_base/model.sdf")
    base.find("model").set("name", "x500_rl_base")
    for sensor in base.findall(".//sensor[@type='imu']"):
        sensor.find("update_rate").text = "1000"
        for axis in sensor.findall("imu/angular_velocity/*"):
            noise = axis.find("noise")
            if noise is None:
                noise = ET.SubElement(axis, "noise", type="gaussian")
            noise.set("type", "gaussian")
            for key, value in {"mean": 0, "stddev": 0.0017453292,
                               "bias_mean": 0, "bias_stddev": 0,
                               "dynamic_bias_stddev": 0}.items():
                element = noise.find(key)
                if element is None:
                    element = ET.SubElement(noise, key)
                element.text = str(value)
    base.write(base_dir / "model.sdf", encoding="unicode", xml_declaration=True)
    mass = sum(float(x.text) for x in base.findall(".//inertial/mass"))
    model_dir = models / "x500_rl"
    model_dir.mkdir(exist_ok=True)
    model = ET.parse(upstream / "x500/model.sdf")
    node = model.find("model")
    node.set("name", "x500_rl")
    node.find("include/uri").text = "model://x500_rl_base"
    motor = node.find("plugin")
    max_speed = float(motor.findtext("maxRotVelocity"))
    kf = float(motor.findtext("motorConstant"))
    slowdown = float(motor.findtext("rotorVelocitySlowdownSim"))
    for other in node.findall("plugin"):
        for key in ["maxRotVelocity", "motorConstant", "rotorVelocitySlowdownSim"]:
            if float(other.findtext(key)) != float(motor.findtext(key)):
                raise RuntimeError("Current collective-thrust model requires identical rotor parameters")
    hover = math.sqrt(mass * 9.8 / (4 * kf)) / max_speed
    plugin = ET.Element("plugin", filename="librate_training_system.so", name="RateTrainingSystem")
    ET.SubElement(plugin, "max_speed").text = str(max_speed)
    ET.SubElement(plugin, "hover").text = str(hover)
    ET.SubElement(plugin, "slowdown").text = str(slowdown)
    # x500_base carries its own model z offset; free-flight height refers
    # to the base_link feedback seen by the controller.
    base_pose = [float(x) for x in base.findtext("model/pose", "0 0 0 0 0 0").split()]
    spawn_height = 5.0 - base_pose[2] if mode == "free_flight" else 3.0
    ET.SubElement(plugin, "spawn_height").text = str(spawn_height)
    node.insert(1, plugin)
    if mode == "ball_rig":
        joint = ET.SubElement(node, "joint", name="training_ball_joint", type="ball")
        ET.SubElement(joint, "parent").text = "world"
        ET.SubElement(joint, "child").text = "base_link"
        # Ideal pivot at base_link origin: no extra fixture inertia/friction.
        ET.SubElement(joint, "pose", relative_to="base_link").text = "0 0 0 0 0 0"
    model.write(model_dir / "model.sdf", encoding="unicode", xml_declaration=True)
    for folder, name in [(base_dir, "x500_rl_base"), (model_dir, "x500_rl")]:
        (folder / "model.config").write_text(f'<model><name>{name}</name><version>1.0</version><sdf version="1.9">model.sdf</sdf></model>')
    world = ET.parse(px4 / "Tools/simulation/gz/worlds/default.sdf")
    wn = world.find("world")
    wn.set("name", "rate_training")
    wn.find("physics/max_step_size").text = "0.001"
    wn.find("physics/real_time_update_rate").text = "1000"
    systems = [("physics", "Physics"), ("user-commands", "UserCommands"),
               ("scene-broadcaster", "SceneBroadcaster"), ("imu", "Imu"),
               ("magnetometer", "Magnetometer"), ("air-pressure", "AirPressure"), ("navsat", "NavSat")]
    for filename, name in systems:
        ET.SubElement(wn, "plugin", filename=f"gz-sim-{filename}-system", name=f"gz::sim::systems::{name}")
    noise_plugin = ET.SubElement(wn, "plugin", filename="libimu_random_walk_system.so", name="ImuRandomWalkSystem")
    ET.SubElement(noise_plugin, "input_topic").text = "/world/rate_training/model/x500_rl_0/link/base_link/sensor/imu_sensor/imu"
    ET.SubElement(noise_plugin, "bias_random_walk").text = "0.0002"
    include = ET.SubElement(wn, "include")
    ET.SubElement(include, "uri").text = "model://x500_rl"
    ET.SubElement(include, "name").text = "x500_rl_0"
    ET.SubElement(include, "pose").text = f"0 0 {spawn_height} 0 0 0"
    world.write(worlds / "rate_training.sdf", encoding="unicode", xml_declaration=True)
    config = dict(px4=str(px4), runtime=str(runtime), px4_commit=revision, mode=mode,
                  world=str(worlds / "rate_training.sdf"), models=str(models),
                  plugin_dir=str(runtime / "gazebo-build"),
                  max_rotor_rad_s=max_speed, mass_kg=mass, thrust_coefficient=kf,
                  gravity_m_s2=9.8, snapshot_version=2,
                  rotor_slowdown=slowdown, control_dt_s=0.001, physics_dt_s=0.001,
                  hover_speed_fraction=hover, hover_thrust_fraction=hover**2)
    config["gyro_noise"] = dict(location="Gazebo IMU before PX4", white_stddev_rad_s=0.0017453292,
                                bias_walk_rad_s_per_sqrt_s=0.0002, initial_bias_rad_s=0,
                                note="Engineering defaults, not measured board calibration; walk runs during simulator warmup too")
    (runtime / f"runtime.{mode}.json").write_text(json.dumps(config, indent=2) + "\n")
    return config


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--px4", type=pathlib.Path, required=True)
    p.add_argument("--runtime", type=pathlib.Path, required=True)
    p.add_argument("--mode", choices=["free_flight"], default="free_flight",
                   help="Generate the current free-flight assets")
    args = p.parse_args()
    px4, runtime = args.px4.resolve(), args.runtime.resolve()
    configs = {mode: prepare_mode(px4, runtime, mode) for mode in ("free_flight",)}
    config = configs[args.mode]
    (PROJECT / "runtime.local.json").write_text(json.dumps(config, indent=2) + "\n")
    (runtime / "runtime.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


if __name__ == "__main__":
    main()
