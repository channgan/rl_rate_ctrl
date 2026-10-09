import numpy as np
import original_missions as old
import lateral_missions as lateral
CASES={**old.CASES,**lateral.CASES}
def mission(name,t):return lateral.mission(name,t) if name in lateral.CASES else old.mission(name,t)
class NativeDiagnosticMission:
 def __new__(cls,name,state,backend):
  if name in lateral.CASES:return lateral.NativeDiagnosticMission(name,state,backend)
  obj=old.NativeDiagnosticMission(name,state,backend);obj.base_world=np.zeros(3);obj.save=lambda:None;return obj
