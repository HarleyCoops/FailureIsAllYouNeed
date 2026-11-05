import copy
import unittest
from pathlib import Path
import numpy as np
from crossgame.asterix import Asterix,Policy,play,diagnose,train
from crossgame.run import protected

BASE=Path(__file__).resolve().parents[1]/'crossgame/assets/asterix-base.npz'


class CrossGameTests(unittest.TestCase):
    def test_native_environment_clone_preserves_shared_rng_and_replay(self):
        env=Asterix(5,1000)
        for _ in range(25):env.step(0)
        branch=copy.deepcopy(env)
        self.assertIs(branch.game.random,branch.game.env.random)
        self.assertIsNot(branch.game.random,env.game.random)
        for action in [1,2,3,4,0]*5:
            if env.done:break
            env.step(action);branch.step(action)
            self.assertEqual(env.snapshot(),branch.snapshot())

    def test_observation_matches_upstream_channel_mapping(self):
        env=Asterix(3)
        for _ in range(20):env.step(0)
        raw=env.game.state();obs=env.observation().reshape(10,10,6)
        for a,b in [(0,0),(1,1),(2,3),(4,2)]:np.testing.assert_array_equal(obs[:,:,a],raw[:,:,b])
        self.assertFalse(obs[:,:,3].any());self.assertFalse(obs[:,:,5].any())

    def test_counterfactual_sequence_replays_from_original_reset(self):
        policy=Policy.load(BASE);episode=play(policy,5,1000,trace=True)
        targets,report=diagnose(policy,episode)
        self.assertTrue(targets)
        self.assertIn(0,report['anchors']);self.assertIn(episode['steps']-1,report['anchors'])
        for t in targets:
            replay=play(policy,5,1000,prefix=episode['actions'][:t['anchor']]+t['actions'])
            self.assertEqual(replay['total'],t['total']);self.assertEqual(replay['steps'],t['steps'])
            self.assertGreater(replay['total'],episode['total'])

    def test_gate_rejects_return_loss_and_lost_survival(self):
        base=[dict(seed=1,total=3.,reason='time_limit')]
        self.assertFalse(protected(base,[dict(seed=1,total=4.,reason='collision')])[0])
        self.assertFalse(protected(base,[dict(seed=1,total=2.,reason='time_limit')])[0])
        self.assertTrue(protected(base,[dict(seed=1,total=4.,reason='time_limit')])[0])

    def test_gradient_update_changes_weights_and_reduces_correction_loss(self):
        policy=Policy.load(BASE);episode=play(policy,5,1000,trace=True)
        targets,_=diagnose(policy,episode)
        candidate,loss=train(policy,targets[:1],episode['observations'][::8],steps=100)
        self.assertNotEqual(candidate.fingerprint(),policy.fingerprint())
        self.assertLess(loss[-1]['ce'],loss[0]['ce'])


if __name__=='__main__':unittest.main()
