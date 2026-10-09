"""Independent NumPy KKT and active-normal audit for both declared points."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from bias_projection_active import constraints
from bias_projection import SOLVER_TOL

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--geometry',type=Path,required=True);ap.add_argument('--admission',type=Path,required=True);a=ap.parse_args()
    admission=json.loads((a.admission/'admission.json').read_text());report={}
    for name in ['pre18','pre40']:
        inp=torch.load(a.geometry/f'{name}_projection_inputs.pt',weights_only=False,map_location='cpu')
        p=inp[0].numpy().astype(float);At,bt,_,_,labels=constraints(*inp);A=At.numpy();b=bt.numpy()
        r=admission['points'][name]['bias_intersection'][0];d=r['projection'];ids=[labels.index(k) for k in d['active_constraints']];aw=A[ids]
        # Independent NumPy least squares for equality-constrained nearest point.
        correction=np.linalg.lstsq(aw,b[ids]-aw@p,rcond=None)[0];x=p+correction
        lam=np.asarray(d['multipliers']);res=A@x-b
        cert=dict(primal=max(0.,float(res.max())),stationarity=float(np.max(np.abs(x-p+A.T@lam))),dual=max(0.,float(-lam.min())),complementarity=float(np.max(np.abs(lam*res))))
        assert max(cert.values())<=SOLVER_TOL
        sv=np.linalg.svd(aw,compute_uv=False)
        old=json.loads((a.geometry/f'{name}_geometry.json').read_text());previous=np.asarray(old['final_delta'])
        report[name]=dict(independent_certificate=cert,active_constraints=d['active_constraints'],active_singular_values=sv.tolist(),active_condition_number=float(sv[0]/sv[-1]),
                          distance_to_old_dykstra_last_iterate=float(np.linalg.norm(x-previous)),
                          old_iterate_max_constraint_violation=max(0.,float((A@previous-b).max())),
                          solution_squared_distance_to_proposal=float(np.sum((x-p)**2)),
                          old_infeasible_iterate_squared_distance_to_proposal=float(np.sum((previous-p)**2)),
                          accepted=r['accepted'],nonzero_action=r['action_delta_max']>0)
    (a.admission/'independent_certificate.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
