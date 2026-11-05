# Asterix: learning from rejected candidates

The first hosted trial repaired targets with 41 candidates, but all failed
preservation. This second experiment treats those regressions as training
counterexamples while retaining the original legal corrective paths.

## Frozen inputs and split roles

The incumbent is still the original Astra-authored Asterix checkpoint: no update
passed the first run. Three prior candidates were selected from its frozen dataset
revision, one per cycle with a qualifying candidate, using only development/gate
results. Each selected candidate improved both means; within its cycle we preferred
fewer per-seed losses and lost survivors. Paths, tensor fingerprints, selection
scores and original corrective sequences are pinned in crossgame/config.json.

Both seed ranges 0–31 and 1000–1031 now explicitly supply training data, including
regression examples. Their original development/gate names remain in metrics for
comparison, but neither is independent evaluation. The previous audit 50000–50127
is excluded. The new audit 70000–70127 runs only after selection ends and is not
downloaded, mined or used as training feedback.

## Feedback mechanism

1. Replay each imported candidate and the incumbent on the paired training seeds.
2. Record every return loss and every lost horizon survivor.
3. Prioritize up to eight worst regression trajectories per rejected candidate.
   Save both full traces, including native RNG state, and verify their common
   world state before the first differing action.
4. Extract the first divergent decision and up to 32 further disagreement states
   along the known better incumbent trajectory. Those actions supply preservation
   labels. We never train a failed action as a positive example simply because it
   occurred in a rejected run.
5. Replay the imported original corrective sequences under the exact native rules
   and add their verified labels alongside the currently targeted fix.
6. Jointly train correction cross-entropy, ordinary KL preservation, a parameter
   displacement penalty, and a targeted action-margin loss on regression examples.
7. Replay every candidate on both training populations, even when it fails its
   target. Add its new regressions to memory before the next proposal.

Identical observations are deduplicated; repeatedly exposed states receive higher
sampling weight. Memory holds at most 4096 states. Labels are refreshed against
the current accepted policy after promotion. Every detected regression is recorded;
the bounded traced subset and memory cap are explicit computational limits.
Conflicting auxiliary corrective paths and exact observation/label conflicts are
reported rather than silently relabelled. The active correction takes precedence
in the loss, but the actual preservation gate is never waived for a conflict.

## Budget and acceptance

Six cycles, up to two target paths, two learning rates, two KL weights, and three
feedback refinements give at most 144 candidate trainings. Each uses 300 AdamW
steps over all network layers. Feedback loss weights are 1, 5 and 20 across the
three refinements. Each proposal starts from accepted weights; the accumulated
memory, not an unverified checkpoint, carries rejected-candidate feedback forward.

The gate stays fixed: correct the target under prefix-conditioned and full-reset
replay, preserve accepted repairs, increase development mean, lose no per-seed
return on either training population, and lose no horizon survivors. Publishing
also requires positive mean gain and zero regressions on the new final audit.
No threshold is relaxed retrospectively. Success would demonstrate an accepted
improvement with regression feedback, not isolate its causal benefit from all
other proposal-budget changes; a matched no-feedback ablation would be needed.

## Hosted outputs

- Dataset: HarleyCooper/asterix-recursive-trajectories-02
- Conditional model: HarleyCooper/asterix-self-improved-02
- Dashboard and native simulator: HarleyCooper/asterix-recursive-learning-02
- W&B: christian-cooper-us/failure-is-all-you-need-crossgame

The original game rules, MinAtar 1.0.15, 0.1 native sticky actions, greedy policy
and 1,000-step horizon are unchanged. Counterfactual replay clones native RNG
state; it does not introduce exploratory action noise. The CPU Job is bounded to
60 minutes. All original source and license provenance remain attached.
