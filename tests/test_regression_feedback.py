import tempfile
import unittest
from pathlib import Path
import numpy as np
from crossgame.asterix import Policy,play,train
from crossgame.feedback import RegressionMemory,compatible_targets
from crossgame.run import write

BASE=Path(__file__).resolve().parents[1]/'crossgame/assets/asterix-base.npz'


class RegressionFeedbackTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base=Policy.load(BASE)
        cls.bad=Policy(cls.base.weights);cls.bad.weights['b3'][1]=10000
        cls.before=play(cls.base,5,1000);cls.after=play(cls.bad,5,1000)

    def test_regression_becomes_correct_teacher_labels_and_exact_evidence(self):
        self.assertGreater(self.before['total'],self.after['total'])
        with tempfile.TemporaryDirectory() as directory:
            memory=RegressionMemory([5],1000,directory,write)
            event=memory.ingest(self.base,self.bad,[self.before],[self.after],'test-rejection')
            x,y,counts=memory.arrays(self.base)
            self.assertGreater(len(x),0)
            self.assertTrue(all(self.base.act(o)==a for o,a in zip(x,y)))
            self.assertTrue(any(self.bad.act(o)!=a for o,a in zip(x,y)))
            self.assertEqual(len(event['all_regressions']),1)
            self.assertTrue(list(Path(directory).glob('*.json.gz')))
            old=len(x);memory.ingest(self.base,self.bad,[self.before],[self.after],'repeat')
            self.assertEqual(len(memory.entries),old)
            self.assertTrue((memory.arrays(self.base)[2]==2*counts).all())

    def test_final_audit_seed_cannot_enter_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            memory=RegressionMemory([0,1],1000,directory,write)
            with self.assertRaisesRegex(ValueError,'outside feedback'):
                memory.ingest(self.base,self.bad,[self.before],[self.after],'forbidden')

    def test_promoted_audit_labels_do_not_copy_failed_incumbent(self):
        with tempfile.TemporaryDirectory() as directory:
            memory=RegressionMemory([5],1000,directory,write)
            memory.ingest(self.base,self.bad,[self.before],[self.after],'old-audit',fixed_labels=True)
            x,y,_=memory.arrays(self.bad)
            self.assertTrue(all(self.base.act(o)==a for o,a in zip(x,y)))
            self.assertTrue(any(self.bad.act(o)!=a for o,a in zip(x,y)))

    def test_recovery_accepts_better_score_at_same_timeout_horizon(self):
        from crossgame.feedback import corrected_outcome
        failure=dict(total=10.,steps=1000,reason='time_limit')
        better=dict(total=12.,steps=1000,reason='time_limit')
        collision=dict(total=12.,steps=900,reason='collision')
        self.assertTrue(corrected_outcome(better,failure,recovery=True))
        self.assertFalse(corrected_outcome(collision,failure,recovery=True))

    def test_known_better_suffix_replays_from_improved_parent(self):
        from crossgame.feedback import recovery_path
        parent=Policy.load(BASE.with_name('continuation-base.npz'))
        failure=play(parent,70008,1000)
        target=recovery_path(self.base,parent,failure,1000)
        self.assertIsNotNone(target)
        replay=play(parent,70008,1000,prefix=failure['actions'][:target['anchor']]+target['actions'])
        self.assertGreater(replay['total'],failure['total'])
        self.assertEqual(replay['reason'],'time_limit')

    def test_conflicting_auxiliary_path_is_disclosed_not_silently_relabelled(self):
        obs=np.zeros((1,600),dtype=np.float32)
        primary=dict(observations=obs,actions=[1]);other=dict(observations=obs,actions=[2])
        selected,conflicts=compatible_targets(primary,[other])
        self.assertEqual(len(selected),1);self.assertEqual(conflicts,1)

    def test_training_consumes_rejection_memory_and_reports_conflicts(self):
        ep=play(self.base,5,1000,trace=True)
        obs=ep['observations'][0];action=(self.base.act(obs)+1)%5
        target=dict(observations=np.asarray([obs]),actions=[action])
        with tempfile.TemporaryDirectory() as directory:
            memory=RegressionMemory([5],1000,directory,write)
            memory.ingest(self.base,self.bad,[self.before],[self.after],'test')
            feedback=memory.arrays(self.base)
            candidate,history=train(self.base,[target],ep['observations'][::8],steps=50,feedback=feedback)
            self.assertGreater(history[-1]['feedback_states'],0)
            self.assertTrue(np.isfinite(history[-1]['regression_margin_loss']))
            self.assertNotEqual(candidate.fingerprint(),self.base.fingerprint())


if __name__=='__main__':unittest.main()
