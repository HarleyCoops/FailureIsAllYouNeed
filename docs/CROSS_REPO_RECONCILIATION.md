# Cross-repository reconciliation — 2026-11-03

The frozen Qwen result and the Kimi analytic-repair results use different policies
and action-selection protocols. Their scores are not a shared leaderboard.

## Qwen: sampled improvement, greedy regression

| Protocol | Baseline | Repaired | Status |
| --- | ---: | ---: | --- |
| Sampled, 1,024 audit games | 1.2802734375 | 4.0478515625 | Original frozen audit |
| Greedy, same 1,024 seeds | 10.2470703125 | 1.5673828125 | Posthoc local diagnostic |

The latter used NumPy 2.4.1, float32, on Windows. It is not a new held-out test,
and greedy decisions near tied logits can depend on numerical details.
The sampled result remains reproducible, but does **not** establish improvement
under greedy deployment. Both probabilities and argmax decisions changed:
all 2,108 observations pooled from baseline/repaired sampled practice trajectories
had different argmax actions (the pool includes repeated observations).

Seed 314159 still misses in both original sampled free-play episodes: return 1,
step 14. The controlled prefix replay also remains unprevented. Under this local
greedy runtime, baseline returns 11 and misses at step 96; repaired returns 0 and
misses at step 6. A reported repaired return of 14 has not been reproduced under
these exact checkpoints and protocols. This is a reconciliation issue, not grounds
to substitute that result into the frozen audit.

The patch changes three scalar parameters. Describing it as a wholesale parameter
replacement is inaccurate. The failure of tested analytic edits does not prove
that this policy is irreparable by another constrained solver.

## Separate subsequent experiments

Kimi's current `cross_repo2.py` uses the first 256 original Qwen audit seeds for
candidate selection and one action RNG shared across games (`default_rng(7)`).
The frozen Qwen protocol instead resets action randomness for every seed using
`SeedSequence([seed, 73021])`. The subsequent search therefore uses a different
protocol; those 256 seeds are development data for it. This does not retroactively
invalidate the original audit, which followed a frozen candidate.

Kimi v1/v3 were evaluated on seeds 20000–20999. The first 64 were previously used
for a pilot evaluation. Seventeen v3 failures from that set informed v4 development;
v4's entire old-set result is labeled development, while its fresh evaluation uses
30000–30999. Zero observed v1 losses is a finite-sample result, not a guarantee.

## Reproduction and evidence

Run `python scripts/compare_action_modes.py` with the repository requirements.
It records runtime, fingerprints, per-seed greedy scores, failure replay, and action
comparison in `results/action-mode-check/summary.json`. This diagnostic never edits weights.
The original sampled evidence remains under `results/qualified-repair-01/`.
[Consolidated dataset](https://huggingface.co/datasets/HarleyCooper/failure-is-all-you-need-evaluations) · [End-of-day status](https://github.com/HarleyCoops/FailureIsAllYouNeed/blob/codex/initial-results/docs/STATUS_2026-11-03.md).
