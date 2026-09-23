"""Record the training episode containing each global step milestone."""
import csv
import json
from pathlib import Path

import gymnasium as gym
import numpy as np


class TrajectoryClock:
    def __init__(self, directory, interval=40_000, *, label='Training episode', stochastic_actions=True):
        self.directory = Path(directory)
        self.interval = interval
        self.steps = 0
        self.label = label
        self.stochastic_actions = stochastic_actions


class EpisodeTrajectory(gym.Wrapper):
    def __init__(self, env, clock):
        super().__init__(env)
        self.clock = clock
        self.native_torque = getattr(self.unwrapped.config, 'native_torque', False)
        self.rows = []
        self.state_rows = []
        self.milestones = []
        self.stage = None
        self.gate_events = []
        self.gate_max_seconds = 0.
        self.gate_observed = False

    def reset(self, **kwargs):
        self._finish('reset_before_terminal', False)
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.clock.steps += 1
        self.stage = info['training_stage']
        gate = info.get('waypoint_gate_transition')
        if gate is not None:
            self.gate_observed = True
            self.gate_max_seconds = max(self.gate_max_seconds,gate['settled_seconds'])
            if gate['switched']:
                self.gate_events.append(dict(gate,training_step=self.clock.steps,
                    episode_time_s=(len(self.rows)+1)*self.unwrapped.config.dt))
        row = [self.clock.steps, info['sim_us'],
            *np.rad2deg(info['target_rad_s']),
            *np.rad2deg(info['rates_true_rad_s']),
            *np.rad2deg(info['rates_rad_s']),
            *np.asarray(info['current_pwm_command'], dtype=float)]
        if self.native_torque:
            torque = np.asarray(info['torque_command'], dtype=float)
            if torque.shape != (3,) or not np.all(np.isfinite(torque)):
                raise ValueError('Expected three accepted requested torque commands for trajectory')
            # Preserve the first 15 legacy columns; append this transition's
            # policy torque, not the preceding observation's torque history.
            row.extend(torque)
        self.rows.append(row)
        if 'position_target_ned' in info:
            state = self.unwrapped.current_state
            q = np.asarray(state['q_ned_frd'],dtype=float)
            q /= np.linalg.norm(q)
            w,x,y,z = q
            yaw = np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z))
            position = np.asarray(info['position_ned'])
            target = np.asarray(info['position_target_ned'])
            self.state_rows.append([self.clock.steps,info['sim_us'],*position,*target,
                np.rad2deg(yaw),np.rad2deg(info['attitude_target_rpy_rad'][2]),
                np.linalg.norm(target-position),gate['settled_seconds'] if gate else 0.])
        if self.clock.steps % self.clock.interval == 0:
            self.milestones.append(self.clock.steps)
        if terminated or truncated:
            outcome = info.get('failure') or ('success' if info.get('is_success') else 'truncated')
            self._finish(outcome, bool(terminated and not truncated))
        return observation, reward, terminated, truncated, info

    def _finish(self, outcome, complete):
        if self.milestones:
            self._save(outcome, complete)
        self.rows = []
        self.state_rows = []
        self.milestones = []
        self.gate_events = []
        self.gate_max_seconds = 0.
        self.gate_observed = False

    def _save(self, outcome, complete):
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg

        self.clock.directory.mkdir(parents=True, exist_ok=True)
        data = np.asarray(self.rows)
        # Each row contains post-action feedback and the reference used for
        # that transition's reward, before the next command is sampled.
        times = np.arange(1, len(data) + 1) * self.unwrapped.config.dt
        fig = Figure(figsize=(11, 19 if self.native_torque else 13), constrained_layout=True)
        FigureCanvasAgg(fig)
        axes = fig.subplots(10 if self.native_torque else 7, 1, sharex=True)
        for i, axis in enumerate(axes[:3]):
            axis.plot(times, data[:, 8+i], color='0.65', linewidth=.6, label='Gyro (noisy)')
            axis.plot(times, data[:, 5+i], linewidth=1, label='True body rate')
            axis.step(times, data[:, 2+i], where='pre', linestyle='--', label='Rate command')
            axis.set_ylabel(('Roll', 'Pitch', 'Yaw')[i] + ' (deg/s)')
            axis.grid(alpha=.25)
        axes[0].legend()
        if self.native_torque:
            from .trajectory_plot import draw_torque_axes
            draw_torque_axes(axes[3:6], times, data[:, 15:18])
        for i, axis in enumerate(axes[-4:]):
            axis.step(times, data[:, 11+i], where='pre', linewidth=.6, color=f'C{i}')
            axis.set_ylabel(f'Allocated PWM {i}' if self.native_torque else f'PWM {i}')
            axis.set_ylim(-.05, 1.05)
            axis.axhline(.001, color='red', linestyle=':', linewidth=.5)
            axis.axhline(.999, color='red', linestyle=':', linewidth=.5)
            axis.grid(alpha=.25)
        axes[-1].set_xlabel('Episode time (s)')
        from .trajectory_plot import apply_axes
        apply_axes(axes, times, data)
        if self.gate_observed:
            # The publisher displays this PNG directly. Draw actual switches
            # here rather than relying on an optional later CSV redraw.
            for event in self.gate_events:
                for axis in axes[:-4] if self.native_torque else axes[:3]:
                    axis.axvline(event['episode_time_s'], color='purple', linestyle=':', linewidth=.8)
            axes[0].text(.01, .97,
                f'Waypoint switches: {len(self.gate_events)}; longest settled interval: '
                f'{self.gate_max_seconds:.2f}s', transform=axes[0].transAxes,
                va='top', fontsize=8, bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
        fig.suptitle(f'{self.clock.label} | {self.stage} | {outcome} | '
                     f'{"complete" if complete else "INCOMPLETE"}\n'
                     f'Steps {int(data[0,0])}-{int(data[-1,0])}; milestones {self.milestones}')
        for milestone in self.milestones:
            base = self.clock.directory / f'step_{milestone:09d}'
            fig.savefig(base.with_suffix('.png'), dpi=130)
            with base.with_suffix('.csv').open('w', newline='') as stream:
                writer = csv.writer(stream)
                columns = ['training_step', 'sim_us', 'target_roll_deg_s', 'target_pitch_deg_s',
                    'target_yaw_deg_s', 'true_roll_deg_s', 'true_pitch_deg_s', 'true_yaw_deg_s',
                    'gyro_roll_deg_s', 'gyro_pitch_deg_s', 'gyro_yaw_deg_s',
                    'pwm_0', 'pwm_1', 'pwm_2', 'pwm_3']
                if self.native_torque:
                    columns += ['torque_roll_normalized', 'torque_pitch_normalized', 'torque_yaw_normalized']
                writer.writerow(columns)
                writer.writerows(self.rows)
            if self.state_rows:
                with base.with_suffix('.state.csv').open('w',newline='') as stream:
                    writer = csv.writer(stream)
                    writer.writerow(['training_step','sim_us','north_m','east_m','down_m',
                        'target_north_m','target_east_m','target_down_m','yaw_deg',
                        'target_yaw_deg','waypoint_distance_m','settled_seconds'])
                    writer.writerows(self.state_rows)
            base.with_suffix('.json').write_text(json.dumps(dict(milestone=milestone,
                first_step=int(data[0,0]), last_step=int(data[-1,0]), complete=complete,
                outcome=outcome, stage=self.stage, stochastic_training_actions=self.clock.stochastic_actions,
                plot_label=self.clock.label,
                duration_s=float(times[-1]),
                trajectory_format_version=2 if self.native_torque else 1,
                torque_source=('current accepted requested normalized FRD policy torque; not measured or allocator-achieved torque'
                               if self.native_torque else None),
                torque_range=[-1, 1] if self.native_torque else None,
                waypoint_switches=self.gate_events if self.gate_observed else None,
                waypoint_max_settled_seconds=self.gate_max_seconds if self.gate_observed else None,
                pwm_source=('current_pwm_command: actual allocated motor commands, diagnostic only'
                            if self.native_torque else
                            'current_pwm_command: clipped normalized PWM actually issued for this transition'),
                pwm_range=[0, 1], motor_indices=[0, 1, 2, 3]), indent=2))

    def close(self):
        try:
            self._finish('environment_closed', False)
        finally:
            self.env.close()
