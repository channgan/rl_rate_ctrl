import copy
import pytest
import torch
import bias_projection_active as candidate
import bias_projection as baseline
import admit_bias_projection as admission
import test_bias_projection as original_tests

@pytest.fixture(autouse=True)
def use_candidate(monkeypatch):
    monkeypatch.setattr(original_tests,'bp',candidate)
    monkeypatch.setattr(admission,'project',candidate.project)

test_declared_synthetic_cases=original_tests.test_declared_synthetic_cases
test_adjacent_float_bias_boundary=original_tests.test_adjacent_float_bias_boundary
test_both_anchors_bind_and_cost_is_not_discarded=original_tests.test_both_anchors_bind_and_cost_is_not_discarded
test_all_zero_gradients_and_zero_proposal=original_tests.test_all_zero_gradients_and_zero_proposal
test_projection_does_not_mutate_inputs_and_replays_exactly=original_tests.test_projection_does_not_mutate_inputs_and_replays_exactly

def test_bounded_failure(monkeypatch):
    monkeypatch.setattr(candidate,'MAX_CYCLES',0)
    with pytest.raises(RuntimeError,match='iteration limit'):
        candidate.project(torch.ones(4),[],torch.zeros(2),torch.zeros(2),torch.zeros(2))

def test_duplicate_and_opposing_constraints_have_known_solution():
    z=torch.zeros(2);p=torch.tensor([1.,2.,0.,0.]);g=torch.tensor([1.,0.,0.,0.])
    d,r=candidate.project(p,[g,g*2,-g],z,z,z)
    assert torch.equal(d,torch.tensor([0.,2.,0.,0.]))
    assert max(r['certificate'].values())<=candidate.SOLVER_TOL

def test_oblique_intersection_analytic_projection():
    z=torch.zeros(2);p=torch.tensor([2.,1.,0.,0.])
    # Projection on x+y<=0 is (.5,-.5); second constraint x>=0 is inactive.
    d,r=candidate.project(p,[torch.tensor([1.,1.,0.,0.]),torch.tensor([-1.,0.,0.,0.])],z,z,z)
    assert torch.allclose(d,torch.tensor([.5,-.5,0.,0.]),atol=1e-7,rtol=0)
    assert max(r['certificate'].values())<=candidate.SOLVER_TOL

def test_cost_scaling_invariance():
    z=torch.zeros(2);p=torch.tensor([2.,1.,.01,-.01]);g=torch.tensor([1.,1.,0.,0.])
    d,_=candidate.project(p,[g],z,z,z)
    for factor in [1e-6,1e6]:
        other,r=candidate.project(p,[g*factor],z,z,z)
        assert torch.equal(d,other)

def test_input_shape_rejected():
    with pytest.raises(ValueError,match='shape'):
        candidate.project(torch.ones(4),[torch.ones(3)],torch.zeros(2),torch.zeros(2),torch.zeros(2))
