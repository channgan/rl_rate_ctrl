"""Readable trajectory display without changing or discarding recorded samples."""
import numpy as np
from matplotlib.ticker import MultipleLocator


def draw_torque_axes(axes, times, torques):
    for i, ax in enumerate(axes):
        ax.step(times, torques[:, i], where='pre', lw=.8, color=f'C{i}')
        ax.set_ylabel(('Roll', 'Pitch', 'Yaw')[i] + ' torque\n(normalized)')
        ax.axhline(0., color='0.6', lw=.5)
        for bound in (-.999, .999):
            ax.axhline(bound, color='red', ls=':', lw=.5)
        extent = float(np.max(np.abs(torques[:, i])))
        limit = float(np.clip(np.ceil(1.1 * extent / .05) * .05, .05, 1.05))
        ax.set_ylim(-limit, limit)
        ax.yaxis.set_major_locator(MultipleLocator(.02 if limit <= .1 else .1 if limit <= .5 else .5))
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(alpha=.25)


def apply_axes(axes, times, data):
    for i, ax in enumerate(axes[:3]):
        values = data[:, 2+i:9+i:3]
        # Keep ordinary tracking visible without letting a few spikes set the scale.
        extent = max(np.max(np.abs(values[:,0])), np.percentile(np.abs(values[:,1:]),99.5))
        limit = float(np.clip(np.ceil(1.1*extent/5)*5,10,100))
        tick = 2 if limit <= 15 else 5 if limit <= 35 else 10
        ax.set_ylim(-limit, limit)
        ax.yaxis.set_major_locator(MultipleLocator(tick))
        ax.yaxis.set_minor_locator(MultipleLocator(tick/5))
        ax.tick_params(axis='y', labelsize=8)
        outside = np.any(np.abs(values) > limit, axis=1)
        if outside.any():
            for j in range(values.shape[1]):
                high, low = values[:, j] > limit, values[:, j] < -limit
                ax.scatter(times[high], np.full(high.sum(), limit*.97), marker='^', color='crimson', s=12)
                ax.scatter(times[low], np.full(low.sum(), -limit*.97), marker='v', color='crimson', s=12)
            ax.text(.99,.97, f'Outside view: {outside.sum()} samples; full range {values.min():.1f} to {values.max():.1f} deg/s',
                    transform=ax.transAxes, ha='right', va='top', fontsize=8,
                    bbox=dict(facecolor='white', alpha=.8, edgecolor='none'))
        ax.grid(which='minor', alpha=.08)
    for ax in axes[-4:]:
        ax.set_ylim(-.02,1.02)
        ax.yaxis.set_major_locator(MultipleLocator(.1))
        ax.tick_params(axis='y', labelsize=8)
    axes[-1].set_xlim(0, max(times[-1], .01))
    axes[-1].xaxis.set_major_locator(MultipleLocator(1 if times[-1] > 5 else .25))


def redraw(csv_path, report):
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    data = np.atleast_2d(np.loadtxt(csv_path, delimiter=',', skiprows=1))
    has_torque = report.get('torque_source') is not None
    if has_torque and data.shape[1] != 18:
        raise ValueError('Native torque trajectory requires 18 columns')
    times = np.arange(1,len(data)+1)*report['duration_s']/len(data)
    fig = Figure(figsize=(12,20 if has_torque else 15), constrained_layout=True)
    FigureCanvasAgg(fig)
    axes = fig.subplots(10 if has_torque else 7,1,sharex=True)
    for i, ax in enumerate(axes[:3]):
        ax.plot(times,data[:,8+i],color='.65',lw=.6,label='Gyro (noisy)')
        ax.plot(times,data[:,5+i],lw=1,label='True body rate')
        ax.step(times,data[:,2+i],where='pre',ls='--',label='Rate command')
        ax.set_ylabel(('Roll','Pitch','Yaw')[i]+' (deg/s)')
    axes[0].legend(loc='lower left',fontsize=8)
    if has_torque:
        draw_torque_axes(axes[3:6], times, data[:,15:18])
    for i, ax in enumerate(axes[-4:]):
        ax.step(times,data[:,11+i],where='pre',lw=.6,color=f'C{i}')
        ax.set_ylabel(f'Allocated PWM {i}' if has_torque else f'PWM {i}')
        for bound in (.001,.999):
            ax.axhline(bound,color='red',ls=':',lw=.5)
    for ax in axes:
        ax.grid(alpha=.25)
    apply_axes(axes,times,data)
    switches = report.get('waypoint_switches')
    if switches is not None:
        for event in switches:
            for ax in axes[:-4] if has_torque else axes[:3]:
                ax.axvline(event['episode_time_s'],color='purple',ls=':',lw=.8)
        axes[0].text(.01,.97,f'Waypoint switches: {len(switches)}; longest settled interval: '
            f"{report.get('waypoint_max_settled_seconds',0):.2f}s",transform=axes[0].transAxes,
            va='top',fontsize=8,bbox=dict(facecolor='white',alpha=.8,edgecolor='none'))
    axes[-1].set_xlabel('Episode time (s)')
    fig.suptitle(f"{report.get('plot_label', 'Training episode')} | {report['stage']} | {report['outcome']} | "
        f"{'complete' if report['complete'] else 'INCOMPLETE'}\n"
        f"Steps {report['first_step']}-{report['last_step']} | Adaptive rate axes (deg/s); red markers indicate overflow")
    output = csv_path.with_suffix('.png')
    temporary = output.with_suffix('.tmp.png')
    fig.savefig(temporary,dpi=130)
    temporary.replace(output)
