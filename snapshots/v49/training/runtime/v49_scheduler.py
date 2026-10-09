"""V48 final guards with one changed proposal solver; no history-stage switches."""
import json
import torch
from adaptive_smooth import AdaptiveScheduler
from bias_projection_active import project
from bias_projection import ProjectionFailure

class V49Scheduler(AdaptiveScheduler):
    def __init__(self,actor,data,config_hash,guard_h,guard_mean,anchor,anchor_mean):
        super().__init__(actor,data,config_hash,guard_h,guard_mean)
        self.anchor_w=anchor.rp_w.detach().clone()
        self.anchor_b=anchor.rp_b.detach().clone()
        self.anchor_mean=anchor_mean.detach().clone()

    @torch.no_grad()
    def diagnostics(self):
        r=super().diagnostics();w=self.actor.rp_w;b=self.actor.rp_b
        r['global_A_weight_relative']=float(((w-self.anchor_w).norm(dim=1)/self.anchor_w.norm(dim=1)).max())
        r['global_A_bias_change']=float(abs(b-self.anchor_b).max())
        mu=torch.clamp(torch.nn.functional.linear(self.guard_h,w,b),-1,1)
        r['global_A_guard_action_change']=float(abs(mu-self.anchor_mean[:,:2]).max())
        r['strict_A_bias']=float((b.double()-self.anchor_b.double()).abs().max())
        r['accepted']=bool(r['accepted'] and r['global_A_weight_relative']<=.01 and r['global_A_bias_change']<=1e-4 and r['global_A_guard_action_change']<=.002 and r['strict_A_bias']<=1e-4-1e-10)
        return r

    def project_proposal(self,proposal,gradients):
        try:
            delta,info=project(proposal,gradients,self.actor.rp_b.detach(),self.initial['rp_b'],self.anchor_b)
        except ProjectionFailure as error:
            self.last_projection=dict(passed=False,error=str(error),**error.diagnostics)
            self.record_projection();raise
        except Exception as error:
            self.last_projection=dict(passed=False,error=repr(error));self.record_projection();raise
        self.last_projection=dict(passed=True,**info);self.record_projection();return delta

    def record_projection(self):
        if getattr(self,'folder',None):
            with (self.folder/'projection_solver.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(before_local_update=self.updates+1,**self.last_projection),allow_nan=False)+'\n')
