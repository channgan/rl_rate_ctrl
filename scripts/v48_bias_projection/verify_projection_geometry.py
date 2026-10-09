"""Independent NumPy Dykstra recurrence audit; no optimizer or native calls."""
from pathlib import Path
import argparse,json
import numpy as np,torch
from bias_projection import LIMIT,SOLVER_TOL
from bias_projection_active import constraints

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--geometry',type=Path,required=True);a=ap.parse_args();out={}
    for name in ['pre18','pre40']:
        inputs=torch.load(a.geometry/f'{name}_projection_inputs.pt',weights_only=False,map_location='cpu')
        p,gs,cur,batch,anchor=inputs
        A,b,lower,upper,labels=constraints(*inputs);A=A.numpy();b=b.numpy();G=np.stack([g.numpy().astype(np.float64) for g in gs]);norm=np.linalg.norm(G,axis=1);N=G/norm[:,None]
        q=json.loads((a.geometry/f'{name}_geometry.json').read_text());x=p.numpy().astype(np.float64);initial=x.copy();cor=np.zeros((10,len(x)))
        for iteration in range(len(q['history'])):
            previous=x.copy()
            for i,g in enumerate(N):
                y=x+cor[i];x=y-max(0.,np.dot(g,y))*g;cor[i]=y-x
            before_box=N@x
            y=x+cor[-1];x=y.copy();x[-2:]=np.maximum(-b[-2:],np.minimum(b[-4:-2],y[-2:]));cor[-1]=y-x
        sv=np.linalg.svd(A,compute_uv=False);rank=int(np.sum(sv>np.finfo(float).eps*max(A.shape)*sv[0]))
        v=dict(numpy_recurrence_delta_max_difference=float(np.max(np.abs(x-np.asarray(q['final_delta'])))),
               dykstra_correction_invariant_residual=float(np.max(np.abs(initial-x-cor.sum(0)))),
               constraint_rank=rank,nonzero_spectrum_condition_number=float(sv[0]/sv[rank-1]),
               zero_singular_values_explained_by='opposing upper/lower normals for each of two biases; not infeasibility',
               cost_normal_singular_values=np.linalg.svd(N,compute_uv=False).tolist(),
               final_cost_residual_before_box=before_box.tolist(),final_cost_residual_after_box=(N@x).tolist(),
               normalized_gradient_scale_range=[float(norm.min()),float(norm.max())],
               current_bias_margin_to_float32_bounds=[b[-2:].tolist(),b[-4:-2].tolist()],
               current_nonlinear_guards_pass=q['current_nonlinear_guards']['accepted'],
               fixed_iterations=len(q['history']),finite=True)
        assert v['numpy_recurrence_delta_max_difference']<1e-15 and v['dykstra_correction_invariant_residual']<1e-15
        assert np.all(b>=0) and np.isfinite(A).all() and np.isfinite(b).all()
        if name=='pre40':
            tail=np.array([r['max_positive_residual'] for r in q['history'][-64:]])
            v['last64_residual_ratio_median']=float(np.median(tail[1:]/tail[:-1]))
            v['threshold_ratios']=dict(feasibility=q['history'][-1]['max_positive_residual']/SOLVER_TOL,change=q['history'][-1]['max_change']/SOLVER_TOL)
        out[name]=v
    (a.geometry/'independent_verification.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))

if __name__=='__main__':main()
