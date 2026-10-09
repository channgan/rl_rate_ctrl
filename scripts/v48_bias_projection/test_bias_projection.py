import pytest
import torch
import bias_projection as bp
from admit_bias_projection import synthetic, same

def test_declared_synthetic_cases():
    assert len(synthetic())==8

@pytest.mark.parametrize('sign',[-1.,1.])
def test_adjacent_float_bias_boundary(sign):
    z=torch.zeros(2)
    inside=torch.tensor(bp.LIMIT,dtype=torch.float32)
    if float(inside)>bp.LIMIT:inside=torch.nextafter(inside,torch.tensor(-float('inf')))
    cur=sign*inside.repeat(2)
    d,r=bp.project(torch.tensor([.1,.1,sign*.1,sign*.1]),[],cur,z,z)
    assert ((cur+d[-2:]).double().abs()<=bp.LIMIT).all()
    outside=torch.nextafter(inside,torch.tensor(float('inf')))
    with pytest.raises(ValueError,match='Infeasible'):
        bp.project(torch.ones(4),[],sign*outside.repeat(2),z,z)

def test_both_anchors_bind_and_cost_is_not_discarded():
    batch=torch.tensor([0.,0.]);anchor=torch.tensor([.00015,-.00015]);cur=torch.tensor([.000075,-.000075])
    g=torch.tensor([1.,0.,1.,0.])
    d,r=bp.project(torch.tensor([1.,-1.,1.,-1.]),[g],cur,batch,anchor)
    target=cur+d[-2:]
    assert ((target.double()-batch.double()).abs()<=bp.LIMIT).all()
    assert ((target.double()-anchor.double()).abs()<=bp.LIMIT).all()
    assert float(g.double().dot(d.double()))<=bp.CAST_TOL
    assert d[1]==-1

def test_all_zero_gradients_and_zero_proposal():
    z=torch.zeros(2);d,_=bp.project(torch.zeros(4),[torch.zeros(4)]*9,z,z,z)
    assert torch.equal(d,torch.zeros(4))

def test_iteration_exhaustion_rejects_without_fallback(monkeypatch):
    monkeypatch.setattr(bp,'MAX_CYCLES',0)
    with pytest.raises(RuntimeError,match='iteration limit'):
        bp.project(torch.ones(4),[],torch.zeros(2),torch.zeros(2),torch.zeros(2))

def test_projection_does_not_mutate_inputs_and_replays_exactly():
    z=torch.zeros(2);p=torch.tensor([1.,2.,3.,4.]);gs=[torch.tensor([1.,1.,0.,0.])]
    saved=[p.clone(),gs[0].clone(),z.clone()]
    a=bp.project(p,gs,z,z,z);b=bp.project(p,gs,z,z,z)
    assert same(a,b) and same(saved,[p,gs[0],z])
