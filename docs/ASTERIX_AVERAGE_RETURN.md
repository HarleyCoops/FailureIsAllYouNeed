# Asterix: recursive learning with an average-return objective

On November 5, 2026, the user explicitly changed the objective: individual game
regressions no longer veto an otherwise useful average improvement. This is a
prospective protocol for run 04 and a disclosed retrospective release decision for
run 03. Run 03 did not pass its original zero-regression publication rule; its
original summary remains immutable evidence of that result.

## Run 03 release

Source: `5597b6758f20ada54686cc250c86fae5b00af361`.
Evidence: `HarleyCooper/asterix-recursive-trajectories-03` revision
`187efee607aeccec859dd9d1d4072a016fe82321`.
Final policy fingerprint: `70c1c9ae42992840ef20fc9352decd9a28f8eab48363e48f14d54f1d88645979`.

Six cycles, 67 proposals, one accepted checkpoint. Development mean rose from
57.76875 to 59.0875; selection-set mean rose from 62.03125 to 62.0625. Across those
192 games there were 28 wins and no return losses or lost survivors versus run 02.
Previously observed original-reference return losses fell from 26 to 9 and lost
survivors from 11 to 6.

The 128-game fresh audit mean rose from 56.0078125 to 56.4375, with 14 wins,
10 losses, 104 ties, and five lost survivors relative to run 02. Against the original
policy, mean rose from 55.8671875 to 56.4375, with 32 wins, 26 losses, 70 ties,
and nine lost survivors. Average gains are measured effects, not proof of general
superiority or statistical significance. Survival losses can overlap return losses.

Release: `HarleyCooper/asterix-self-improved-03`. A separate release decision records
the new objective without rewriting the original summary's `publishable: false`.
The simulator loads the published weights at a pinned revision and verifies their
fingerprint. It must show the known regressions, not imply universal preservation.

## Run 04 protocol

- Start from exactly the published run 03 final checkpoint.
- Training: 288 development games (0–31, 70000–70127, 90000–90127) and 32 selection
  games (1000–1031). All are development data for this experiment.
- Import the prior 618-state memory as fixed saved teacher labels, and replay all
  former-audit return/survival regressions using run 02 as the reference teacher.
  The original policy is still separately evaluated for historical comparisons.
- Preserve game physics, observation interface, greedy inference, native 0.1 sticky
  actions, and 1,000-step horizon.
- Six cycles with the existing bounded candidate optimizer/search settings.
- A candidate must correct its target and increase the pooled mean across all 320
  training games. Do not average the two group means with equal weight.
- Individual score losses, lost survivors, and prior-repair losses are reported but
  do not block acceptance. Record regressions from accepted AND rejected proposals.
  Better teacher actions remain fixed when the incumbent advances, rather than being
  relabeled to a newly accepted but worse action on that case.
- Retain the finite replay-memory budget and disclosed per-candidate trace cap. This
  is not exhaustive causal coverage; conflicting targets remain a research limit.
- Replay every accepted checkpoint under the same mean-return rule before release.
- Final audit: 512 fresh games, seeds 110000–110511, evaluated after selection ends.
  Never feed them into this run's training. A later reuse retires their audit status.
- Publication requires an accepted update, exact replay, and positive fresh-audit
  mean gain over the starting run 03 checkpoint. No individual-loss or survivor veto.
  Report original-reference comparisons, per-game losses, and uncertainty separately.

## Infrastructure

Git source and config are pinned in the container. Run 04 uses dataset
`HarleyCooper/asterix-recursive-trajectories-04`, Space
`HarleyCooper/asterix-recursive-learning-04`, and conditional model
`HarleyCooper/asterix-self-improved-04`. W&B remains in
`christian-cooper-us/failure-is-all-you-need-crossgame` with a distinct run ID.
The bounded CPU job retains the existing 60-minute limit. Durable cycle snapshots
preserve weights, candidates, traces, memory, and selection receipts.

## Interpretation

Optimizing expected return is a reasonable objective. The risk is repeated trading
of one failure for another, catastrophic tails hidden by the mean, or overfitting
to increasingly reused development games. Keep tracking those outcomes and compare
against an unchanged original policy. Do not interpret passing a positive-mean gate
as proving statistical significance, universal improvement, or the benefit of
recursion over a matched control. No physics or inference-time solver is changed.
