"""Read-only tensor observations; never consume RNG or modify optimizer tensors."""
import hashlib
import torch

def raw(s,loss,ix,ratio,adv):
    if not getattr(s,'telemetry_enabled',True):return
    norms=torch.stack([p.grad.detach().norm(2) for p in s.actor.parameters()]);n=float(norms.norm(2))
    s.step_telemetry=dict(minibatch_clipped_loss=float(loss.detach()),minibatch_unclipped_loss=float(-(s.minibatch_weights*ratio.detach()*adv).sum()),raw_gradient_norm=n,gradient_clip_factor=min(1.,.5/(n+1e-6)),sample_index_sha256=hashlib.sha256(ix.cpu().numpy().tobytes()).hexdigest())

def projected(s,proposal,delta):
    if not getattr(s,'telemetry_enabled',True):return
    s.step_telemetry.update(adam_proposal_norm=float(proposal.detach().norm()),projected_norm=float(delta.detach().norm()),projection_change_norm=float((proposal-delta).detach().norm()),active_constraints=s.last_projection['active_constraints'])

@torch.no_grad()
def full(s):
    lp=s.likelihood()[0];ratio=torch.exp(lp-s.data['behavior']);adv=s.data['adv'];w=s.data['weights']
    return dict(clipped_loss=float(-(w*torch.minimum(ratio*adv,ratio.clamp(.95,1.05)*adv)).sum()),unclipped_loss=float(-(w*ratio*adv).sum()))
