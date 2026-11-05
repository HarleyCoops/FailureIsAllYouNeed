# Asterix run 03: continue the improved policy and learn from its audit failures

Starting point: run 02 final.npz, fingerprint
9193a27ca10bceafe415cc4a1abe5d6220e622cc28c5e70e6f9cdd44104a6f3a.
The pinned parent dataset revision and every imported seed appear in config.json.
This is the four-update improved research checkpoint, not the original network.
Its four accepted repairs and 378 regression-memory states are inherited.

## Former audit becomes training

Run 02 had 26 return regressions and 11 lost survivors on seeds 70000–70127.
All 11 survival failures overlap the 26 return regressions: there are 26 distinct
failed cases. Every one is replayed and traced, with no eight-case sampling cap
on this initial import. The initial counterexamples use the original network's
known better trajectories. These labels are fixed: refreshing them from the
currently failing improved policy would accidentally teach its errors back.

The full former audit joins the training population. Preserving its other cases
protects previously discovered gains. Training now covers 192 seeds: original
development 0–31, original gate 1000–1031, and former audit 70000–70127.
None of these is independent evaluation. The fresh audit is 90000–90127, used
only after selection ends. Prior audit 50000–50127 remains excluded.

## Recursive correction

The loop prioritizes unrecovered former-audit failures. For a known better
reference trajectory, it locates the first differing decision and verifies the
complete legal reference suffix from the actual common prefix. That provides a
multi-action corrective training path, including return regressions in episodes
that already reach the time limit. It never changes physics or random state.
If no suitable reference path exists, bounded reachable-state diagnosis remains
available. Native MinAtar 1.0.15 rules, sticky actions 0.1, greedy action selection
and the 1,000-step horizon remain unchanged.

Candidate optimization jointly trains the correction, inherited accepted
behavior, and regression memory. New rejected proposals contribute additional
counterexamples as in run 02. The six-cycle budget, 300 optimizer steps per
candidate, learning rates and preservation weights are frozen before launch.

Every accepted update must improve training return, preserve every training
seed's incumbent return and horizon survival, and retain all inherited repairs.
Recovery paths can improve score at the same horizon; surviving beyond a
1,000-step timeout is not an impossible requirement imposed on these cases.
Progress can be retained incrementally without fixing all 26 in one proposal.

## Release checks

The final checkpoint is compared with both the improved parent and the original
network. Publication requires additional accepted improvement, positive fresh
audit mean gain over the parent, zero fresh return/survival losses versus either
reference, and no remaining original-reference losses on the 192 training seeds.
An improved candidate that still fails these checks remains available as a
research checkpoint for the next recursive experiment, without being advertised
as an audit-passing release.

Dataset: HarleyCooper/asterix-recursive-trajectories-03.
Conditional model: HarleyCooper/asterix-self-improved-03.
Dashboard and paired inference: HarleyCooper/asterix-recursive-learning-03.
W&B project: christian-cooper-us/failure-is-all-you-need-crossgame.
The CPU Job uses a frozen GitHub source commit and a 60-minute limit.
