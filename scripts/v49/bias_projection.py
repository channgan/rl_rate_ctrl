"""Offline candidate only: fixed Dykstra projection, no guard relaxation."""
import torch

LIMIT = 1e-4 - 1e-10
MAX_CYCLES = 512
SOLVER_TOL = 1e-12
CAST_TOL = 1e-10  # normalized linear residual, not a nonlinear guard allowance

class ProjectionFailure(RuntimeError):
    def __init__(self,message,diagnostics):
        super().__init__(message)
        self.diagnostics=diagnostics

def project(proposal, gradients, current, batch, anchor):
    tensors = [proposal, current, batch, anchor, *gradients]
    if not all(torch.isfinite(t).all() for t in tensors):
        raise ValueError('Nonfinite projection input')
    lower = torch.maximum(batch.double()-LIMIT, anchor.double()-LIMIT)
    upper = torch.minimum(batch.double()+LIMIT, anchor.double()+LIMIT)
    if (lower > upper).any():
        raise ValueError('Empty bias intersection')
    if ((current.double() < lower) | (current.double() > upper)).any():
        raise ValueError('Infeasible current bias')
    # Round absolute bounds inward to representable actor values, before solving.
    lo = lower.float(); hi = upper.float()
    lo = torch.where(lo.double()<lower, torch.nextafter(lo, torch.full_like(lo,float('inf'))), lo)
    hi = torch.where(hi.double()>upper, torch.nextafter(hi, torch.full_like(hi,-float('inf'))), hi)
    if (lo > hi).any():
        raise ValueError('No representable bias intersection')
    lb=lo.double()-current.double(); ub=hi.double()-current.double()
    gs=[]
    for g in gradients:
        g=g.detach().double(); n=g.norm()
        if n>1e-24: gs.append(g/n)
    x=proposal.detach().double().clone()
    corrections=[torch.zeros_like(x) for _ in range(len(gs)+1)]
    converged=False
    for cycle in range(MAX_CYCLES):
        prev=x.clone()
        for i,g in enumerate(gs):
            y=x+corrections[i]; x=y-torch.clamp(g.dot(y),min=0)*g; corrections[i]=y-x
        y=x+corrections[-1]; x=y.clone(); x[-2:]=torch.maximum(lb,torch.minimum(ub,y[-2:])); corrections[-1]=y-x
        residual=max([0.]+[float(g.dot(x)) for g in gs])
        if residual<=SOLVER_TOL and float((x-prev).abs().max())<=SOLVER_TOL:
            converged=True; break
    if not converged:
        raise ProjectionFailure('Fixed projection iteration limit exhausted',dict(
            cycles=MAX_CYCLES,normalized_residual=max([0.]+[float(g.dot(x)) for g in gs]),
            last_cycle_change=float((x-prev).abs().max()) if MAX_CYCLES else None,
            raw_cost_dot=[float(g.double().dot(x)) for g in gradients],
            lower=lower.tolist(),upper=upper.tolist(),
            bias_box_residual=max(0.,float((lb-x[-2:]).max()),float((x[-2:]-ub).max())),
            converged=False))
    delta=x.to(proposal.dtype)
    target=current+delta[-2:]
    cast_residual=max([0.]+[float(g.dot(delta.double())) for g in gs])
    if cast_residual>CAST_TOL or ((target.double()<lower)|(target.double()>upper)).any():
        raise RuntimeError('Projection cast violated feasibility')
    return delta, dict(cycles=cycle+1, normalized_residual=residual,
                      cast_normalized_residual=cast_residual,
                      raw_cost_dot=[float(g.double().dot(delta.double())) for g in gradients],
                      bias_target=target.tolist(), lower=lower.tolist(), upper=upper.tolist())
