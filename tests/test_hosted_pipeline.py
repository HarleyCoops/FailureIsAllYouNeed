import unittest

import numpy as np
import torch

from experiment09 import loop as e
from hosted_pipeline.gradient import TrainablePolicy, train_candidate
from hosted_pipeline.job import publishable


class HostedPipelineTests(unittest.TestCase):
    def fixture(self):
        from weight_space_breakout.policy import SHAPES
        rng = np.random.default_rng(70)
        return e.Policy({k:rng.normal(0, .05, shape).astype(np.float32) for k,shape in SHAPES.items()})

    def test_exact_tensor_roundtrip_and_matching_forward(self):
        p = self.fixture()
        network = TrainablePolicy(p)
        self.assertEqual(e.policy_hash(p), e.policy_hash(network.to_policy()))
        obs = e.Breakout().observation()
        np.testing.assert_allclose(network(torch.from_numpy(obs)).detach().numpy(), p.logits(obs), atol=1e-6)

    def test_backprop_training_changes_weights_and_reduces_correction_loss(self):
        p = self.fixture()
        obs = e.Breakout().observation()
        target = dict(observation=obs, action=(p.act(obs, None, "greedy")+1)%5)
        probes = []
        for case in e.witnesses()[:30]:
            game = e.Breakout()
            game.reset(case["seed"])
            probes.append(game.observation())
        q, report = train_candidate(p, target, np.asarray(probes), "adam-high", 1., steps=40)
        self.assertNotEqual(e.policy_hash(p), e.policy_hash(q))
        self.assertLess(report["history"][-1]["correction_ce"], report["history"][0]["correction_ce"])
        self.assertEqual(report["optimizer"], "AdamW")
        self.assertEqual(len(report["parameter_tensors_trained"]), 6)
        self.assertGreater(report["history"][0]["gradient_norm"], 0)

    def test_publication_requires_audited_improvement_without_regressions(self):
        summary = dict(final_mean=10., baseline_mean=8., regressed=0, accepted_checkpoints=1)
        self.assertTrue(publishable(summary, {"status":"passed"}))
        for change in ({"regressed":1}, {"final_mean":8.}, {"accepted_checkpoints":0}):
            self.assertFalse(publishable({**summary, **change}, {"status":"passed"}))
        self.assertFalse(publishable(summary, {"status":"failed"}))


if __name__ == "__main__":
    unittest.main()
