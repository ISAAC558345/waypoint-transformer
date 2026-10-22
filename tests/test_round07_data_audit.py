"""Focused safety and boundary checks for the non-learning Round-7 tooling."""
from __future__ import annotations
import hashlib, json, tempfile, unittest
from pathlib import Path
import numpy as np
from wt.round7.round7_runner import _case, _difficulty, _quality, _records, _rollout, _row, _specs, _splits, _stitch, _waypoint, radial_projection

ROOT=Path(__file__).resolve().parents[1]
def sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()

class Round7DataAuditTests(unittest.TestCase):
    def test_01_frozen_round_hashes_are_unchanged(self):
        self.assertEqual(sha(ROOT/'round_06'/'ROUND_LOCK.json'),'E77A83EE903739AB6EB327238E632BAEB8FE93E7298258429BA471295AA71772')
        self.assertEqual(sha(ROOT/'round_05'/'ROUND_LOCK.json'),'50124434E0CC9B2B4A3D954A27C97AA139F498C8AE023DA11DE602E533FDD380')
    def test_02_projection_is_legal_and_finite(self):
        for x in (np.array([2.,2.]),np.array([.1,-.2]),np.array([-3.,.5])):
            y=radial_projection(x);self.assertTrue(np.isfinite(y).all());self.assertLessEqual(np.linalg.norm(y),1.+1e-12)
    def test_03_same_scenario_behaviours_share_snapshot_and_trace(self):
        sid,cfg,_,tid,_=_case(2000,3,24);a,ma=_rollout(cfg,sid,2000,tid,_specs()[0],0);b,mb=_rollout(cfg,sid,2000,tid,_specs()[1],1)
        self.assertEqual(a.arrival_trace_id,b.arrival_trace_id);self.assertTrue(np.array_equal(a.uav_positions_m[0],b.uav_positions_m[0]));self.assertTrue(np.array_equal(a.buoy_positions_m[0],b.buoy_positions_m[0]));self.assertTrue(np.array_equal(a.raw_queue_bits[0],b.raw_queue_bits[0]));self.assertFalse(ma['uses_future_information']);self.assertFalse(mb['uses_future_information'])
    def test_04_behaviour_seed_is_reproducible_and_noisy_actions_are_safe(self):
        sid,cfg,_,tid,_=_case(2001,3,24);spec=next(x for x in _specs() if x['behavior_generator']=='noisy_nearest' and x['noise_level']==.25);a,ma=_rollout(cfg,sid,2001,tid,spec,4);b,mb=_rollout(cfg,sid,2001,tid,spec,4)
        self.assertEqual(ma['behavior_seed'],mb['behavior_seed']);self.assertTrue(np.array_equal(a.actions,b.actions));self.assertTrue(np.all(np.linalg.norm(a.actions,axis=1)<=1.+1e-12));self.assertFalse(ma['uses_future_information'])
    def test_05_quality_is_outcome_derived_and_split_is_case_level(self):
        eps=[];meta=[]
        for seed in range(2000,2004):
            for n in (3,5):
                sid,cfg,_,tid,_=_case(seed,n,24)
                for i,s in enumerate(_specs()[:4]):e,m=_rollout(cfg,sid,seed,tid,s,i);eps.append(e);meta.append(m)
        rows=[_row(e,next(m for m in meta if m['episode_id']==e.episode_id)) for e in eps];cases=_difficulty(eps,rows);_quality(rows,cases);splits=_splits(cases)
        for sid in {e.scenario_id for e in eps}:self.assertIn(splits[sid],{'train','validation','test','holdout'})
        for seed in {e.seed for e in eps}:self.assertEqual(len({splits[e.scenario_id] for e in eps if e.seed==seed}),1)
        self.assertEqual({r['quality_tier'] for r in rows},{'high','medium','low'})
        self.assertTrue(all(r['quality_tier']!='low' or np.isfinite(r['episode_return']) for r in rows))
    def test_06_stitching_uses_cross_episode_neighbors_and_no_fabricated_transition(self):
        eps=[];meta=[]
        for seed in range(2000,2003):
            sid,cfg,_,tid,_=_case(seed,3,24)
            for i,s in enumerate(_specs()[:4]):e,m=_rollout(cfg,sid,seed,tid,s,i);eps.append(e);meta.append(m)
        rows=[_row(e,next(m for m in meta if m['episode_id']==e.episode_id)) for e in eps];cases=_difficulty(eps,rows);_quality(rows,cases);rmap={r['episode_id']:r for r in rows};near,cov,graph,stress=_stitch(_records(eps,rmap,stride=8))
        self.assertTrue(all(x['episode_id']!=x['neighbor_episode_id'] for x in near));self.assertTrue(all(x['bridge_candidate_count']>=0 for x in cov));self.assertEqual(stress['selection_rule'].count('fabricated'),1);self.assertEqual(graph[0]['graph_node_count'],len({x['episode_id'] for x in near}))
    def test_07_waypoint_targets_stay_inside_episodes_and_are_finite(self):
        sid,cfg,_,tid,_=_case(2000,3,24);e,m=_rollout(cfg,sid,2000,tid,_specs()[0],0);row=_row(e,m);row['difficulty_tier']='easy';row['quality_tier']='high'
        with tempfile.TemporaryDirectory() as tmp:
            manifest,stats,metadata=_waypoint(Path(tmp)/'samples.npz',[e],{e.episode_id:row},{sid:'train'})
            self.assertEqual(manifest['schema_version'],'waypoint_sample_schema_v2');self.assertTrue(all(r['t']+r['H_w']<=e.episode_length for r in metadata));self.assertTrue(np.isfinite(np.load(Path(tmp)/'samples.npz',allow_pickle=False)['raw_future_delta']).all());self.assertEqual([r['H_w'] for r in stats],[5,10,20,30]);self.assertTrue(all(r['valid_sample_count']>0 for r in stats if r['H_w']<=20))

if __name__=='__main__':unittest.main()
