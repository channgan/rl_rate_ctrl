"""Synthetic zero-native contract tests; no native launcher imported."""
import copy,unittest
from continuation_contract import *

def rows(j=.1):
    return [dict(case=c,seed=s,J=j,yaw=.1,motor=.001,safe=True,axes={f'{r}_{a}_{p}':.1 for r in ('native','requested') for a in ('roll','pitch') for p in ('full','first3','cruise')}) for c in CASES for s in MONITOR_SEEDS]

class ContractTests(unittest.TestCase):
    def test_superseded_entry_disabled(self):
        with self.assertRaises(RuntimeError):prepare('/not_accessed','/not_written')
    def test_native_million_block(self):
        p=block_plan('native_scoring_1ms');self.assertEqual(p['native_scoring_per_block'],1003520);self.assertEqual(p['optimizer_updates_per_block'],70);self.assertEqual(p['total_ceiling'],11301000);self.assertEqual(p['inside_block_high_cost_monitor_jobs'],0)
    def test_legacy_million_block(self):
        p=block_plan('legacy_outer_10ms');self.assertEqual(p['legacy_outer_per_block'],1003520);self.assertEqual(p['optimizer_updates_per_block'],700);self.assertEqual(p['total_ceiling'],73482000)
    def test_unspecified_unit_and_unbounded_rejected(self):
        with self.assertRaises(AssertionError):block_plan(None)
        with self.assertRaises(AssertionError):block_plan('native_scoring_1ms',4)
    def test_budget_includes_baseline(self):
        js=plan();self.assertEqual(sum(x['cap'] for x in js),8340000);self.assertEqual(len(js),174)
        self.assertTrue(all(x['kind']=='monitor' and x['stage']==0 for x in js[:9]))
        self.assertEqual(sum(x['kind']=='training' for x in js),84)
        self.assertEqual(sum(x['kind']=='monitor' for x in js),36)
    def test_full_cap_settlement(self):
        b=Budget()
        for j in plan():b.reserve(j);b.settle(j['id'],j['cap'])
        self.assertEqual(b.charged,LIMIT);self.assertFalse(b.reserved)
    def test_failed_attempt_charged_no_retry(self):
        b=Budget();j=plan()[0];b.reserve(j);b.settle(j['id'],123,failed=True)
        self.assertEqual(b.charged,123)
        with self.assertRaises(AssertionError):b.reserve(j)
        with self.assertRaises(AssertionError):b.reserve(plan()[1])
    def test_overrun_retained(self):
        b=Budget();j=plan()[0];b.reserve(j)
        with self.assertRaises(AssertionError):b.settle(j['id'],47001)
        self.assertTrue(b.stopped);self.assertEqual(b.charged,47001)
    def test_changed_cap_rejected(self):
        j=dict(plan()[0],cap=48000)
        with self.assertRaises(AssertionError):Budget().reserve(j)
    def test_seed_collision(self):
        validate_seed_partition([490301,490302,12701,12702])
        with self.assertRaises(AssertionError):validate_seed_partition([520101])
    def test_missing_baseline(self):
        with self.assertRaises(AssertionError):monitor_decision([],rows(),rows(.09))
    def test_improvement(self):self.assertEqual(monitor_decision(rows(),rows(),rows(.09)),'improvement')
    def test_insufficient_not_convergence(self):
        self.assertEqual(monitor_decision(rows(),rows(),rows(.099)),'insufficient_evidence')
        self.assertEqual(stage_action(2,'insufficient_evidence',1)[0],'terminal_final_only')
    def test_axis_regression_not_hidden(self):
        cur=rows(.09)
        for r in cur:r['axes']['native_pitch_full']=.12
        self.assertEqual(monitor_decision(rows(),rows(),cur),'protection_failure')
    def test_secondary_reference_protected(self):
        cur=rows(.09)
        for r in cur:r['axes']['requested_roll_cruise']=.12
        self.assertEqual(monitor_decision(rows(),rows(),cur),'protection_failure')
    def test_one_seed_regression(self):
        cur=rows(.08)
        for r in cur:
            if r['seed']==520103:r['J']=.11
        self.assertEqual(monitor_decision(rows(),rows(),cur),'insufficient_evidence')
    def test_yaw_and_esc(self):
        for key,value in [('yaw',.12),('motor',.00106)]:
            cur=rows(.09);cur[0][key]=value
            self.assertEqual(monitor_decision(rows(),rows(),cur),'protection_failure')
    def test_near_floor(self):self.assertEqual(monitor_decision(rows(.004),rows(.004),rows(.003)),'insufficient_evidence')
    def test_stage_cap_and_failure(self):
        self.assertEqual(stage_action(3,'improvement',0)[0],'terminal_final_only')
        self.assertEqual(stage_action(1,'protection_failure',0)[0],'stop_diagnose_no_final')
    def test_freshness(self):
        root=Path('/synthetic_continuation')
        m=[dict(result=str(root/'jobs'/f'train_0_{p}'/'result.json'),seed=610101+p,model_id=4142227144,audit_passed=True,sync_passed=True,steps=2048,physical_failure=None) for p in range(7)]
        new_batch_manifest(m,root,0,4142227144)
        m[0]['result']='/historical/result.json'
        with self.assertRaises(AssertionError):new_batch_manifest(m,root,0,4142227144)

if __name__=='__main__':unittest.main()
