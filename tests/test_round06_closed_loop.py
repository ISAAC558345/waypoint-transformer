"""Round-6 evaluation-only invariants; no formal rollout is started here."""
from __future__ import annotations
import hashlib
import json
import unittest
from pathlib import Path
import numpy as np
from wt.maritime.config import MaritimeEnvConfig
from wt.maritime.round3_runner import _build_case_config
from wt.round6.closed_loop import ROOT, R4_CHECKPOINT, R5, WAYPOINT_SHA, _achieved_goal, _reset, _snapshot, _state_from_observation

class Round6ClosedLoopTests(unittest.TestCase):
    def test_01_locked_hashes_exist(self):
        self.assertEqual(hashlib.sha256(R4_CHECKPOINT.read_bytes()).hexdigest().upper(), WAYPOINT_SHA)
        self.assertTrue((ROOT/'round_05'/'ROUND_LOCK.json').is_file())
        self.assertTrue((R5/'selected_transformer_config.json').is_file())

    def test_02_evaluation_seeds_are_new(self):
        cfg=json.loads((ROOT/'round_06'/'configs'/'closed_loop_eval_v1.json').read_text())
        self.assertFalse(set(cfg['scenario_seeds']) & set(range(10)))
        self.assertEqual(len(cfg['scenario_seeds']),30)

    def test_03_snapshot_exact_reset(self):
        case,trace,trace_id=_build_case_config(MaritimeEnvConfig(),scenario_id='closed_loop_eval_v1',scenario_seed=1000,num_buoys=3,horizon=8)
        snap=_snapshot(case,1000,trace,trace_id,'closed_loop_eval_v1'); _,a,ia=_reset(snap); _,b,ib=_reset(snap)
        self.assertTrue(np.array_equal(a,b)); self.assertTrue(np.array_equal(ia['buoy_positions_m'],ib['buoy_positions_m']))
        self.assertEqual(ia['arrival_trace_id'],ib['arrival_trace_id'])

    def test_04_current_state_has_locked_dimensions(self):
        case,trace,trace_id=_build_case_config(MaritimeEnvConfig(),scenario_id='closed_loop_eval_v1',scenario_seed=1001,num_buoys=5,horizon=8)
        _,obs,info=_reset(_snapshot(case,1001,trace,trace_id,'closed_loop_eval_v1')); state,current,nodes=_state_from_observation(obs,info)
        self.assertEqual(state.shape,(62,)); self.assertEqual(current.shape,(13,)); self.assertEqual(nodes.shape,(5,))

    def test_05_goal_is_currently_recomputable(self):
        case,trace,trace_id=_build_case_config(MaritimeEnvConfig(),scenario_id='closed_loop_eval_v1',scenario_seed=1002,num_buoys=3,horizon=8)
        _,_,info=_reset(_snapshot(case,1002,trace,trace_id,'closed_loop_eval_v1')); goal=_achieved_goal(info)
        self.assertEqual(goal.shape,(13,)); self.assertTrue(np.all(np.isfinite(goal))); self.assertTrue(np.all(goal>=-1e-6))

    def test_06_round6_is_evaluation_only(self):
        cfg=json.loads((ROOT/'round_06'/'configs'/'closed_loop_eval_v1.json').read_text())
        self.assertFalse(cfg['training_performed']); self.assertFalse(cfg['waypoint_training_performed']); self.assertFalse(cfg['transformer_training_performed']); self.assertTrue(cfg['closed_loop_evaluation_performed'])

if __name__=='__main__': unittest.main()
