import unittest
from crossgame.selection import accepts, publishable


def row(seed, total, reason='time_limit'):
    return dict(seed=seed, total=total, reason=reason)


class AverageSelectionTests(unittest.TestCase):
    def test_mean_can_improve_with_individual_loss_and_lost_survivor(self):
        before=dict(development=[row(1,10),row(2,10)],gate=[row(3,10)])
        after=dict(development=[row(1,20),row(2,5,'collision')],gate=[row(3,9)])
        self.assertTrue(accepts(before,after,'mean_return')[0])
        self.assertFalse(accepts(before,after,'strict')[0])

    def test_pooled_mean_weights_examples_not_group_means(self):
        before=dict(development=[row(1,10),row(2,10)],gate=[row(3,10)])
        after=dict(development=[row(1,12),row(2,12)],gate=[row(3,7)])
        self.assertTrue(accepts(before,after,'mean_return')[0])
        self.assertFalse(accepts(before,before,'mean_return')[0])
        self.assertFalse(accepts(after,before,'mean_return')[0])

    def test_publication_requires_actual_gain_and_verified_replay(self):
        s=dict(accepted_checkpoints=1,held_out_audit=dict(gain=.5),final_replay_passed=True)
        self.assertTrue(publishable(s,'mean_return',False,False))
        self.assertFalse(publishable(s,'strict',False,False))
        for field,value in [('accepted_checkpoints',0),('final_replay_passed',False),('held_out_audit',dict(gain=0))]:
            self.assertFalse(publishable({**s,field:value},'mean_return'))


if __name__=='__main__':unittest.main()
