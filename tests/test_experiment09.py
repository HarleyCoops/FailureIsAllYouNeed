import tempfile
import unittest
from pathlib import Path

import numpy as np

from experiment09 import loop as e


class RepairLoopTests(unittest.TestCase):
    def policy(self):
        from weight_space_breakout.policy import tracking_policy
        return tracking_policy()

    def test_all_legal_resets_and_reproducible_greedy_play(self):
        cases = e.witnesses()
        self.assertEqual(len(cases), 96)
        self.assertEqual(len({tuple(c["config"]) for c in cases}), 96)
        p = self.policy()
        self.assertEqual(e.play(p, 12), e.play(p, 12))

    def test_no_edit_replays_failure_and_is_not_a_repair(self):
        p = self.policy()
        episode = next(ep for ep in e.census(p, e.witnesses(), trace=True) if ep["reason"] == "miss")
        failure = e.case_from_episode(episode, e.policy_hash(p))
        replay = e.play(p, failure["seed"], prefix=failure["prefix"])
        self.assertEqual(replay, failure["episode"])
        self.assertFalse(e.rescued(replay, failure))
        report, _ = e.verify_candidate(p, p, failure, [], [], [], [])
        self.assertEqual(report["reason"], "no_change")

    def test_jacobian_matches_finite_difference(self):
        rng = np.random.default_rng(51)
        from weight_space_breakout.policy import SHAPES
        p = e.Policy({k: rng.normal(0, .05, shape).astype(np.float32) for k, shape in SHAPES.items()})
        obs = e.Breakout().observation()
        for layer in ("W2", "W3"):
            A, margins = e.margin_system(p, obs, 3, layer)
            direction = rng.normal(size=A.shape[1])
            direction /= np.linalg.norm(direction)
            q = e.Policy(p.weights)
            epsilon = .01
            if layer == "W2":
                q.weights[layer] += (epsilon * direction.reshape(32, 32)).astype(np.float32)
            else:
                update = epsilon * direction.reshape(33, 5)
                q.weights["W3"] += update[:32].astype(np.float32)
                q.weights["b3"] += update[32].astype(np.float32)
            _, after = e.margin_system(q, obs, 3, layer)
            np.testing.assert_allclose((after - margins) / epsilon, A @ direction, atol=2e-4)

    def test_solver_constrains_every_competitor(self):
        A = np.array([[1., -1., 0.], [1., 0., -1.]])
        required = np.array([2., 7.])  # defeating the old winner alone is insufficient
        delta, solved = e.dual_projection(A, required, lambda x: x)
        self.assertTrue(solved)
        self.assertTrue(np.all(A @ delta >= required - 1e-6))

    def test_sequence_diagnosis_rescues_exact_ball_without_mutating_policy(self):
        p = self.policy()
        original_hash = e.policy_hash(p)
        episode = e.play(p, 34, trace=True)
        self.assertEqual(episode["reason"], "miss")
        failure = e.case_from_episode(episode, original_hash)
        targets, _ = e.diagnose_sequence(p, episode, failure)
        self.assertTrue(targets)
        for t in targets:
            prefix = episode["actions"][:t["sequence_start"]] + t["sequence_actions"]
            trial = e.play(p, 34, prefix=prefix)
            self.assertTrue(e.rescued(trial, failure))
            self.assertGreater(trial["total"], episode["total"])
            A, margins = e.target_system(p, t, "W3")
            self.assertEqual(len(margins), 4 * len(t["sequence_actions"]))
        self.assertEqual(e.policy_hash(p), original_hash)
        self.assertEqual(e.play(p, 34), e.serial_episode(episode))

    def test_world_search_inspects_entire_history_and_reachable_states(self):
        p = self.policy()
        episode = e.play(p, 34, trace=True)
        failure = e.case_from_episode(episode, e.policy_hash(p))
        targets, stats = e.diagnose_worlds(p, episode, failure, [episode])
        self.assertEqual(stats["anchors_examined"], list(range(episode["steps"])))
        self.assertGreater(episode["steps"], 12)
        self.assertTrue(targets)
        for t in targets:
            game = e.Breakout()
            game.reset(34)
            prefix = episode["actions"][:t["sequence_start"]] + t["sequence_actions"]
            for action in prefix:
                game.step(action)
            self.assertEqual(e.state(game), t["desired_world"])
            trial = e.play(p, 34, prefix=prefix)
            self.assertTrue(e.world_corrected(trial, failure))

    def test_world_correction_does_not_require_same_ball_at_same_step(self):
        failure = dict(world_target=True, target_step=40, baseline_return=5.)
        self.assertTrue(e.rescued(dict(total=10., steps=80, reason="miss"), failure))
        self.assertTrue(e.rescued(dict(total=24., steps=30, reason="cleared"), failure))
        self.assertFalse(e.rescued(dict(total=10., steps=30, reason="miss"), failure))

    def test_mean_gain_cannot_hide_a_regression(self):
        before = [dict(total=5., reason="miss", actions=[0]), dict(total=5., reason="miss", actions=[0])]
        after = [dict(total=4., reason="miss", actions=[3]), dict(total=20., reason="miss", actions=[4])]
        gate = e.preservation(before, after, before)
        self.assertGreater(gate["after_mean"], gate["before_mean"])
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["losses"], [0])

    def test_failed_old_repair_blocks_candidate(self):
        p = self.policy()
        episode = next(ep for ep in e.census(p, e.witnesses()) if ep["reason"] == "miss")
        failure = e.case_from_episode(episode, e.policy_hash(p))
        failure["accepted_return"] = episode["total"]
        self.assertEqual(e.check_bank(p, [failure]), [failure["id"]])

    def test_recovery_after_commit_before_state_and_tamper_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            p = self.policy()
            q = e.Policy(p.weights)
            q.weights["b3"][0] += .1
            e.save_policy(root / "initial.npz", p)
            e.save_policy(root / "candidate.npz", q)
            plan = dict(cycles=2, initial_policy_hash=e.policy_hash(p))
            receipt = dict(cycle=0, incumbent_hash=e.policy_hash(p), selected_hash=e.policy_hash(q),
                           selected_checkpoint="candidate.npz", accepted=True, bank_entry={"id": "retained"})
            e.write_json(root / "cycle-00/commit.json", receipt)
            current, history = e.recover(root, plan)
            self.assertEqual(current["next_cycle"], 1)
            self.assertEqual(current["checkpoint"], "candidate.npz")
            self.assertEqual(current["repair_bank"], [{"id": "retained"}])
            e.save_policy(root / "candidate.npz", p)
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                e.recover(root, plan)


if __name__ == "__main__":
    unittest.main()
