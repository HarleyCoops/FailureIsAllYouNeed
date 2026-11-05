# Cross-game recursive learning: Asterix

This experiment transfers the failure-diagnosis/training/replay method to another
game from the original RikiyaT/lm_understand-logs release. It does not transfer
Breakout weights. The starting policy is the release's Astra-authored Asterix
network, not a PPO checkpoint. Its 600-32-32-6 architecture has 20,486 parameters.

## Provenance and rules

Upstream source is pinned at 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f.
The original full-precision network and attribution are in crossgame/assets.
MinAtar 1.0.15 supplies the actual environment. Its Asterix source was checked
against the exact source embedded in the upstream prompt and matches.
Weights and upstream experiment materials are CC BY 4.0; the included MinAtar
source in the prompt is GPL-3.0. License texts and hashes accompany the assets.

The player collects gold and avoids enemies; speed and spawn rate increase.
Official native sticky actions remain at 0.1. No additional action noise is added.
The policy is greedy, and the pilot horizon is 1,000 steps. Upstream scores used
sampled actions and 10,000 steps; their numbers are not our baseline.

## Hypothesis and fixed budget

Can simulator-validated corrective sequences become gradient-training targets
that improve a new game's released policy through successive accepted updates?

The run has six cycles, at most two targets and twelve candidate trainings per
cycle, 300 AdamW steps per candidate. All layers can change. Cross-entropy on the
corrective path is balanced with KL preservation on development observations and
a small parameter-displacement penalty. Learning rates and KL weights are frozen
in crossgame/config.json before launch.

Every collision saves its observed trajectory, actions, timers, entity positions,
sticky-action history and random-generator state. Diagnosis searches four-action
branches at up to 24 evenly spaced nodes across the entire trajectory, including
its first and last node. Beam width is six; continuation budget is at most 96.
This is bounded coverage, not exhaustive search of every possible path. Search
can intervene far earlier than the final action. Clones preserve the shared native
random generator, and no branch edits a world state or changes the seed.

A diagnostic correction must earn higher return and outlive the original failure.
Candidate weights must produce that improvement both from the original reset
and when evaluated after the unchanged historical prefix. Prior accepted repair
returns and survival checks remain protected. Failed candidates are recorded.

## Selection and independent final audit

- Development seeds: 0–31. Used for failure mining, preservation examples and selection.
- Gate seeds: 1000–1031. Reused for selection, hence also development evidence.
- Final audit seeds: 50000–50127. Evaluated once after checkpoint selection ends.

Each accepted update must increase development mean return, lose no per-seed
return on either development group, and turn no horizon survivor into a collision.
The final selected model is independently replayed before publication. Publishing
requires at least one accepted update, positive final-audit mean gain, zero
final-audit return losses, and no lost final-audit survivors. The final audit is
not used to choose among intermediate checkpoints. A failed publication audit
leaves the candidate as a research artifact, not a released improved model.

Unlike the Breakout reset census, these seeds are not an exhaustive enumeration
of possible games. Asterix generates random entities during play. Saved RNG state
makes an episode reproducible; the finite audit does not prove universal benefit.

## Hosted pipeline

- Dataset: HarleyCooper/asterix-recursive-trajectories-01
- Conditional model release: HarleyCooper/asterix-self-improved-01
- Dashboard and inference: HarleyCooper/asterix-recursive-learning-01
- W&B project: christian-cooper-us/failure-is-all-you-need-crossgame

The CPU Hugging Face Job uses a Docker Space image built from a pinned GitHub
commit. It uploads candidate evidence, diagnoses, accepted checkpoints, raw
trajectories, split scores and final audit. The Space serves actual native Asterix
inference and loads an improved checkpoint only after a verified release exists.
The run is bounded to 60 minutes. Results, including zero accepted updates, must
be published as observed rather than described as a success in advance.

Local checks: run unittest discovery for test_crossgame.py. Run the experiment
with python -m crossgame.run --output <new-directory>; --hosted enables HF/W&B.
