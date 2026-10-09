"""Deterministic feasible active-set Euclidean projection; offline candidate v2.

The feasible zero increment is a starting point, never an automatic accepted update.
All original nonlinear/nonzero scheduler guards remain outside this solver.
"""
import torch
from bias_projection import LIMIT, MAX_CYCLES, SOLVER_TOL, CAST_TOL, ProjectionFailure

def constraints(proposal,gradients,current,batch,anchor):
    if proposal.ndim!=1 or proposal.numel()<2 or any(g.shape!=proposal.shape for g in gradients):
        raise ValueError('Invalid proposal/gradient shape')
    if any(t.shape!=(2,) for t in [current,batch,anchor]):raise ValueError('Invalid bias shape')
    if not all(torch.isfinite(t).all() for t in [proposal,*gradients,current,batch,anchor]):raise ValueError('Nonfinite projection input')
    lower=torch.maximum(batch.double()-LIMIT,anchor.double()-LIMIT)
    upper=torch.minimum(batch.double()+LIMIT,anchor.double()+LIMIT)
    if (lower>upper).any():raise ValueError('Empty bias intersection')
    if ((current.double()<lower)|(current.double()>upper)).any():raise ValueError('Infeasible current bias')
    lo=lower.float();hi=upper.float()
    lo=torch.where(lo.double()<lower,torch.nextafter(lo,torch.full_like(lo,float('inf'))),lo)
    hi=torch.where(hi.double()>upper,torch.nextafter(hi,torch.full_like(hi,-float('inf'))),hi)
    if (lo>hi).any():raise ValueError('No representable bias intersection')
    lb=lo.double()-current.double();ub=hi.double()-current.double()
    gs=[];labels=[]
    for i,g in enumerate(gradients):
        g=g.detach().double();norm=g.norm()
        if norm>1e-24:gs.append(g/norm);labels.append(f'cost_{i}')
    eye=torch.eye(proposal.numel(),dtype=torch.float64,device=proposal.device)[-2:]
    A=torch.stack(gs+[eye[0],eye[1],-eye[0],-eye[1]])
    b=torch.cat([torch.zeros(len(gs),dtype=torch.float64,device=proposal.device),ub,-lb])
    if (b<0).any():raise ValueError('Infeasible representable current bias')
    labels+=['bias_roll_upper','bias_pitch_upper','bias_roll_lower','bias_pitch_lower']
    return A,b,lower,upper,labels

def project(proposal,gradients,current,batch,anchor):
    A,b,lower,upper,labels=constraints(proposal,gradients,current,batch,anchor)
    p=proposal.detach().double();x=torch.zeros_like(p);working=[];trace=[];converged=False
    for iteration in range(MAX_CYCLES):
        toward=p-x
        if working:
            # Solve A_W.T lambda ~= p-x by SVD; never square its condition number.
            aw=A[working]
            multipliers=torch.linalg.lstsq(aw.T,toward,driver='gelsd').solution
            direction=toward-aw.T@multipliers
        else:
            multipliers=torch.empty(0,dtype=torch.float64);direction=toward
        entry=dict(iteration=iteration+1,working=[labels[i] for i in working],direction_max=float(direction.abs().max()),
                   max_positive_residual=max(0.,float((A@x-b).max())))
        trace.append(entry)
        if float(direction.abs().max())<=SOLVER_TOL:
            if not working or float(multipliers.min())>=-SOLVER_TOL:
                converged=True;entry['action']='KKT candidate';break
            drop=int(torch.argmin(multipliers));entry['action']='drop';entry['constraint']=labels[working[drop]];working.pop(drop)
            continue
        directional=A@direction;slack=b-A@x;alpha=1.;blocker=None
        for i in range(len(b)):
            if i in working or float(directional[i])<=0:continue
            step=max(0.,float(slack[i]/directional[i]))
            if step<alpha:alpha=step;blocker=i
        x=x+alpha*direction
        entry.update(action='move',step=alpha,blocker=None if blocker is None else labels[blocker])
        if blocker is not None:working.append(blocker)
        if not torch.isfinite(x).all() or float((A@x-b).max())>SOLVER_TOL:
            raise ProjectionFailure('Active-set feasibility lost',dict(trace=trace,normalized_residual=float((A@x-b).max())))
    if not converged:raise ProjectionFailure('Fixed active-set iteration limit exhausted',dict(cycles=MAX_CYCLES,trace=trace))
    lam=torch.zeros(len(b),dtype=torch.float64)
    if working:lam[working]=multipliers
    residual=A@x-b
    certificate=dict(primal=max(0.,float(residual.max())),stationarity=float((x-p+A.T@lam).abs().max()),
                     dual=max(0.,float(-lam.min())),complementarity=float((lam*residual).abs().max()))
    if max(certificate.values())>SOLVER_TOL:raise ProjectionFailure('KKT certificate failed',dict(certificate=certificate,trace=trace))
    delta=x.to(proposal.dtype);target=current+delta[-2:];cast=A@delta.double()-b
    if float(cast.max())>CAST_TOL or ((target.double()<lower)|(target.double()>upper)).any():
        raise ProjectionFailure('Projection cast violated feasibility',dict(cast_residual=cast.tolist(),trace=trace))
    return delta,dict(cycles=iteration+1,solver='feasible_active_set_SVD',certificate=certificate,
                     normalized_residual=max(0.,float(residual.max())),cast_normalized_residual=max(0.,float(cast.max())),
                     signed_normalized_residual=residual.tolist(),cast_signed_normalized_residual=cast.tolist(),
                     raw_cost_dot=[float(g.double().dot(delta.double())) for g in gradients],
                     lower=lower.tolist(),upper=upper.tolist(),bias_target=target.tolist(),
                     active_constraints=[labels[i] for i in working],multipliers=lam.tolist(),trace=trace)
