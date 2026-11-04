# Experiment 09: executable numerical pilots

Frozen implementation specification, November 4, 2026. This file specializes
EXPERIMENT_09.md for the first run; the proposed three-arm comparison is future
work. No hosted LLM, action noise, new physics, new reset states, or model
publication is part of this pilot. This is parameter optimization of the small
PPO game policy, executed on the local CPU.

## Inputs and method

- Start: Kimi v4, source NPZ SHA-256
  `87c2248300f93f1bb395f0466602d3616edcf6125ea513a15803d5a84ef3ea40`.
- Local provenance: `artifacts/eod-sync-2026-11-03/KimiBrainSurgery/best_weights_v4.npz`
  in the parent workspace. Copy initial weights into the run directory; never
  overwrite the source checkpoint. Its tensor hash is independently recorded.
- Reuse the existing vendored game and policy. Freeze their source hashes,
  optimizer source, Python version, NumPy version, and configuration in plan.json.
- Greedy float32 policy, first-argmax tie breaking, 200 steps. Enumerate all
  6 paddle centers x 8 ball columns x 2 horizontal velocities using real reset
  seed witnesses. Uniform weighting is over configurations, not witness seeds.
- Every cycle saves every observed miss, including complete world-state history
  and actions, under its checkpoint-specific identity. A seed derived from
  experiment seed 9042026 and cycle number selects an unattempted failure case.
  Randomness selects work; it does not modify play.
- Diagnose up to the last 12 pre-miss decisions after the last successful paddle
  contact. Run 01 tested individual alternative legal actions and found no rescue
  in three selected cases, so produced zero parameter proposals. Its original
  protocol and source are preserved in its source directory.
- Run 02 uses protocol v2: search short sequences of legal actions in cloned games,
  merging identical full world states and retaining at most 64 states per depth.
  Prefer fewer deviations from incumbent decisions, then higher counterfactual
  episode return. At most two saving sequences become parameter targets. Supervise
  all decisions in each sequence, with margins against every competitor. After
  the catch, the incumbent completes each diagnostic episode. These interventions
  are diagnostic only. Failure to find a sequence is not a proof of infeasibility.
- Run 02 was stopped after one committed cycle, with no accepted edit, following
  the user's correction that diagnosis must consider alternative world states
  across the full trajectory. Partial next-cycle artifacts, if any, are not a
  completed trial. Its source, original plan, and stop status remain saved.
- Run 03 uses protocol v3: seed a reachable-state search at EVERY pre-action node
  of the original failure trajectory, including earlier paddle contacts. Search
  breadth-first across those anchors, then up to six legal transitions per branch,
  with at most 2,048 transition expansions and 512 incumbent continuation rollouts.
  Merge identical complete world states. No last-12-steps restriction applies.
  Rank corrective paths by fewest changed decisions, observations shared with other
  failure cases, smallest incumbent logit gaps, and improved return. Shared
  observations are a prioritization heuristic, not proof of shared corrections.
  Retain two hypotheses with different anchors when available. Failure to find a
  path under these budgets does not establish that none exists.
- Store each reachable alternative world, its generating legal path, and its
  incumbent continuation outcome. Estimate parameter changes to make the policy
  follow the corrective path. The resulting policy may find a different path;
  exact replay, not exact imitation of the target world, decides acceptance.

## Parameter estimates

For each target, test penalties 0, 10, 100, 1000, 10000 with W3-plus-bias and W2
edits. At most 20 candidates per cycle. All proposals start from the incumbent.
Use all-four-competitor margin constraints (margin 0.1), not just the old argmax.
The preservation metric uses up to 2,048 evenly selected incumbent observations.

W3 edits minimize a quadratic norm plus logit-change penalty. W2 edits use the
parameter Jacobian and a block metric penalizing hidden-activation changes, with
up to eight nonlinear refinements. Each step has L2 norm at most 2. The metric is
a proposal heuristic; it does not certify behavior preservation. A bounded dual
solver estimates constrained corrections. Numerical nonconvergence is recorded.
All candidates are serialized and reconstructed as validated float32 policies.

## Exact acceptance

1. No-edit control must reproduce the entire saved failed trajectory exactly.
2. In v1/v2, require a catch at the same miss step and column after the frozen
   last-contact prefix. In v3, freeze only the original prefix before the selected
   causal anchor. Allow changed subsequent physics trajectories under the same
   engine: the policy must survive beyond the original miss step (or clear the
   board earlier) AND achieve strictly higher full-episode return.
3. In v3, the candidate's free episode from reset must independently meet both
   the survival/clear and strict return-improvement requirements. Changing one
   selected action or reaching a suggested intermediate state alone is not enough.
4. Every retained repair must still pass its conditional replay and retain at
   least the full-episode return recorded when that repair was first accepted.
5. All 96 free episodes must have return at least that of both v4 and the current
   incumbent. Any previously cleared board must still clear. Changed trajectories
   are counted and saved; identical decisions are not required everywhere.

In v3, every other natural failure corrected by the same edit is also retained
in the repair bank with a reset replay and its accepted return. Report accepted
checkpoints separately from corrected failure cases; one checkpoint may fix more
than one case. At least three accepted checkpoints are needed for pilot success.

The first passing candidate is committed. Others cannot update the incumbent.
This pilot deliberately uses the full census as a reusable development constraint.
It supersedes the earlier proposed one-independent-gate-per-cycle arrangement:
there is no independent hidden selection set in this finite deterministic setup.
The final evaluation is a separate recomputation, not evidence of transfer.

Run for at most ten cycles, stopping after three cycles without acceptance. Save
every diagnosis, candidate, rejection, exact gate trajectory, checkpoint receipt,
and retained repair. An atomic cycle receipt precedes the state cache; resumption
reconstructs accepted history from receipts and verifies policy/source hashes.
An interrupted uncommitted cycle is deterministically recomputed. Do not run two
processes against the same output directory.

Success remains at least three retained repairs and a strictly higher census
mean with zero per-configuration regressions. A negative result is reported as
such, without changing budgets or acceptance rules mid-run.

## Run and inspect

From the FailureIsAllYouNeed directory, with its requirements installed:

```powershell
python -m unittest discover -s tests -p test_experiment09.py -v
python -m experiment09.loop --checkpoint ../artifacts/eod-sync-2026-11-03/KimiBrainSurgery/best_weights_v4.npz --output results/experiment09-run03
python -m experiment09.loop --output results/experiment09-run03 --resume
```

Inspect `summary.json`, `final-census.json`, `state.json`, `failures/`, and each
`cycle-*/commit.json` and `attempt-*.json` under the output directory. Proposal
replay success is separate from accepted checkpoint progress. The comparison
arms and broader problem setups require subsequent separately frozen runs.
