"""Explicit selection objectives; regressions remain evidence under either rule."""
from crossgame.asterix import compare


def accepts(before, after, objective='strict'):
    if objective not in ('strict', 'mean_return'):
        raise ValueError('Unknown selection objective')
    comparisons = {name: compare(before[name], after[name]) for name in before}
    combined = compare(before['development'] + before['gate'], after['development'] + after['gate'])
    if objective == 'mean_return':
        return combined['gain'] > 0, comparisons
    return (all(c['losses'] == 0 and c['lost_survivors'] == 0 for c in comparisons.values())
            and comparisons['development']['gain'] > 0), comparisons


def publishable(summary, objective='strict', historical_pass=True, audit_pass=True):
    if objective not in ('strict', 'mean_return'):
        raise ValueError('Unknown selection objective')
    positive = summary['accepted_checkpoints'] > 0 and summary['held_out_audit']['gain'] > 0
    if objective == 'mean_return':
        return positive and summary['final_replay_passed']
    return positive and historical_pass and audit_pass
