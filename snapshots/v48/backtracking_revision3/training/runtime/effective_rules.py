from pathlib import Path
import json,numpy as np
def enforce_coverage(coverage,entries,batch,model_id,seeds):
 memory=np.asarray([e['memory'] for e in entries]);velocity=np.asarray([e['velocity'] for e in entries])
 assert np.isfinite(memory).all() and np.max(abs(memory))<=1, 'memory finite/bound guard failed'
 assert np.isfinite(velocity).all(), 'velocity finite guard failed'
 m=next(x for x in coverage if x['field']=='memory');v=next(x for x in coverage if x['field']=='velocity')
 diagnostic=dict(batch=batch,model_id=model_id,seeds=seeds,range_min=m['minimum'],range_max=m['maximum'],lower_tail_gap=np.maximum(0,np.asarray(m['minimum'])-m['required']['q05']).tolist(),upper_tail_gap=np.maximum(0,np.asarray(m['required']['q95'])-m['maximum']).tolist(),original_gate_would_fail=not m['passed'],diagnostic_only=True,no_cumulative_current_policy_coverage_claim=True)
 assert v['passed'], 'original velocity coverage gate failed'
 return diagnostic
