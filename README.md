# FailureIsAllYouNeed

**From observed failures to tested parameter repairs.**

A failure trace gives us a place to look: the observations, decisions, and state
changes that led to an unwanted outcome. Our research question is:

> Given a failed trajectory, what small change to the model's parameters would
> change that outcome while preserving performance elsewhere?

We work backward from behavior, propose an explicit weight edit, replay and
evaluate it, and keep it only if it passes a predefined acceptance check.
The long-term aim is a repeatable **observe → diagnose → edit → verify** loop.

This repository publishes our first completed **tool-assisted, verifier-accepted
Breakout repair**. It establishes improved average performance after a sparse
LLM-proposed edit. Preventing the particular diagnostic failure remains an open
objective: that miss was **not** prevented in this run.

[Live repaired policy](https://huggingface.co/spaces/HarleyCooper/weight-space-breakout-qualified-repair-demo-01)
· [Published weights](https://huggingface.co/HarleyCooper/weight-space-breakout-qualified-repair-01/tree/44bef3f4fe96d48b9d67f308d7734f4693846b91)
· [Completed HF Job](https://huggingface.co/jobs/HarleyCooper/6aea9774c78863f9c6322027)
· [W&B run](https://wandb.ai/christian-cooper-us/weight-space-breakout/runs/7a31ec5872dd3f77)

## The result

A Qwen editor inspected and proposed changes to a deliberately damaged,
handwritten game policy. The policy has **17,253 parameters** in a
`500 → 32 → 32 → 5` tanh network. Qwen's own weights were unchanged.

| Measurement | Games | Baseline mean return | Repaired mean return | Paired gain | Paired SE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Practice qualification, used during search | 32 | 1.8750 | 4.4688 | +2.5938 | 0.7788 |
| Independent acceptance gate | 512 | 1.2305 | 3.7813 | **+2.5508** | 0.1954 |
| Independent final audit | 1,024 | 1.2803 | 4.0479 | **+2.7676** | 0.1362 |

![Final independent audit: mean return and per-game outcomes](assets/final-audit.png)

The final audit contained **671 improvements, 243 regressions, and 110 ties**.
Acceptance concerned average return. It did not guarantee improvement on each
seed, and neither policy cleared the board in these 1,024 audit games.

The accepted patch changed **3 parameters (about 0.0174%)**:

| Parameter | Before | After |
| --- | ---: | ---: |
| `W3[0,3]` | +8 | −8 |
| `W3[0,4]` | −8 | +8 |
| `b3[0]` | −3 | 0 |

The first two changes reverse the sign of the hidden feature's contribution to
the left/right action logits. The bias change increases the no-op logit relative
to its previous value. These are direct consequences of the network equations;
the separate causal contribution of each change to the return gain has not been
established by an ablation study.

The editor used **33 model turns**: four inspection calls and 29 proposal calls.
Twenty-eight proposals were valid; one was rejected for an overlong diagnosis.
The first proposal that met the practice rule ended editing automatically.
Its independent gate passed, and the accepted weights were published. The live
Space reproduced the verifier's saved episode exactly. Two additional fresh
inference checks scored 7 and 9; those examples are deployment checks, not another
performance estimate.

## How the experiment worked

```mermaid
flowchart LR
    A[Faulty policy and recorded failure] --> B[Qwen inspects and proposes weight edits]
    B --> C[32 practice games]
    C -->|Does not qualify| B
    C -->|First qualifying edit freezes| D[512 independent gate games]
    D -->|Pass| E[Select repaired weights]
    D -->|Fail| F[Retain baseline]
    E --> G[1024 final audit games]
    F --> G
    G --> H[Publish selected policy and evidence]
```

- Every proposal began from the same faulty baseline and contained at most 64
  scalar assignments. Edits did not accumulate during search.
- Qualification and acceptance each required an actual weight change, mean gain
  greater than 0.10, and gain greater than twice the paired standard error.
- Practice seeds were `4040000–4040031`; gate seeds were `4060000–4060511`;
  final audit seeds were `4090000–4091023`. Gate and audit results were not
  available through the editor's tools.
- The diagnostic replay used seed `314159`. Rescuing it was **not an acceptance
  requirement** in this frozen protocol. Both the controlled replay and the full
  deployed episode still ended in a miss.
- Prime Inference supplied `qwen/qwen3.6-35b-a3b`. HF Jobs CPU ran the existing
  environment, tools, verifier, and final audit. There was no editor-turn cap;
  the rollout had a 60-minute operational timeout.

The prior Prime-hosted launch failed before supplying a trace. That service
failure is preserved in the evidence, with no invented performance score.

## What “working backward” means

The target is an **inverse problem**: find a parameter change consistent with a
desired behavioral correction and preservation constraints. Usually many edits
could change a decision, so a failure does not identify a unique correct patch.

Gradient descent already uses errors to compute parameter updates through
backpropagation. Our distinction is the proposal mechanism: an LLM reads weights
and failure evidence, writes a bounded patch, and receives practice feedback.
This run used no gradient computation or training loop. We are not reconstructing
or reversing the model's training trajectory. This particular starting policy
was handwritten and has no gradient-descent history to reverse.

Tensor shapes describe the architecture; weight values, activations, and
trajectories provide evidence about behavior. A sparse successful intervention
does not by itself prove that a behavior is uniquely located in those parameters.

See [the research idea](docs/IDEA.md) and [the exact protocol](docs/PROTOCOL.md).

## Reproduce the published result

Python 3.11+ and a small CPU are sufficient. No API key, GPU, hosted job, or new
editor call is needed to verify the saved result.

```bash
git clone https://github.com/HarleyCoops/FailureIsAllYouNeed.git
cd FailureIsAllYouNeed
python -m pip install -r requirements.txt
python scripts/reproduce.py
```

The default command checks file hashes, both policy fingerprints, the exact
three-parameter patch, per-seed scores, reported statistics, and the acceptance
decision. To replay all 32 practice, 512 gate, and 1,024 audit games for both
policies with the pinned simulator:

```bash
python scripts/reproduce.py --replay
```

Replaying frozen weights is deterministic in the pinned environment. Generating a
new Qwen conversation is stochastic and is a new experiment, not a guarantee of
the same patch. The original hosted job script and full conversation are included
as provenance; the original launch script uses account-specific paths and is not
a portable one-click launcher.

## Evidence map

| Artifact | What it records |
| --- | --- |
| [Summary](results/qualified-repair-01/summary.json) | Metrics, hashes, seeds, execution identity, and scope |
| [Paired scores](results/qualified-repair-01/paired_scores.csv) | Every baseline/repaired score for all three splits |
| [Exact patch](results/qualified-repair-01/patch.json) | Accepted scalar assignments |
| [Full trace](results/qualified-repair-01/trace-raw.json) | Qwen messages, tool calls, and scoring record |
| [Verifier report](results/qualified-repair-01/verifier-report.json) | Practice qualification, gate, audit, and failure replay |
| [Frozen plan](results/qualified-repair-01/plan.frozen.json) | Rules recorded before the HF execution |
| [Checkpoints](checkpoints/) | Original and accepted NumPy-compatible safetensors |
| [Deployment proof](results/qualified-repair-01/latest-deployment-check.json) | Pinned live checkpoint and exact replay match |
| [Source provenance](vendor/SOURCE.json) | Original environment commit and file hashes |

## What remains to demonstrate

This is one completed repair of a planted fault. It does not establish sustained
recursive improvement, generalization to new game rules, or successful repair of
Cree or Dakota language models. The held-out games test fresh seeds in the same
simulator and distribution. Repeatedly using those seeds for future selection
would make them development data; future trials need new reserved evaluations.

The next scientific target is specific: **prevent a recorded failure under a
controlled replay, then pass a separate preservation test**. Any claim that an
edit isolated a causal mechanism also needs comparisons such as component
ablations and control edits. More checkpoints alone would not establish that.

## Credit and provenance

This is an independent conceptual replication attempt inspired by
[Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/)
by **Rikiya Takehi, Jayden Teoh, Alex Zhang, and Omar Khattab**.
[Original experiment logs](https://github.com/RikiyaT/lm_understand-logs/tree/main/exp1_which_game)
· [Original announcement](https://x.com/rikiyatakehi/status/2117313742891327994).

Our simulator, handwritten policy, planted fault, editor, and tool-assisted
evaluation differ from their experiment. These are our own results, not a rerun
of their original PPO checkpoint or a reproduction of their reported scores.
The name of this repository expresses a research direction; a failure still
needs context, an objective, and independent verification to become a useful
repair signal.

Environment source is preserved from
[`weight-space-breakout-pipeline@0f4697c`](https://github.com/HarleyCoops/weight-space-breakout-pipeline/tree/0f4697c6c8cbfa48e96220c4e59881f5edfbd0b5).
Published November 3, 2026 (America/Winnipeg); raw execution records retain UTC.
Code and project artifacts are provided under the [MIT license](LICENSE), with
the upstream attribution retained. Original third-party reference weights are
not bundled.
