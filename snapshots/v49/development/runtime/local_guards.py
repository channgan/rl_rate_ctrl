import math
def guard_next_native(used,ticks,cap):
 if not all(type(x) is int for x in [used,ticks,cap]) or used<0 or ticks<=0 or cap<=0:raise RuntimeError('invalid native count')
 if used+ticks>cap:raise RuntimeError('internal native budget reached')
def guard_finite(values):
 if not all(math.isfinite(float(x)) for x in values):raise RuntimeError('nonfinite execution value')
