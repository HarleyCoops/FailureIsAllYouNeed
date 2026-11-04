# Published Breakout trial: qualified-repair-v1

The environment was `weight-space-breakout@0.4.5`. Its source is copied unchanged
under `vendor/`, pinned to commit `0f4697c6c8cbfa48e96220c4e59881f5edfbd0b5`.
The original package and full hosted workflow remain in the
[upstream repository](https://github.com/HarleyCoops/weight-space-breakout-pipeline/tree/0f4697c6c8cbfa48e96220c4e59881f5edfbd0b5).

## Frozen decisions

| Item | Value |
| --- | --- |
| Initial policy | Handwritten teaching MLP with reversed left/right controls |
| Parameters | 17,253 float32 values, including biases |
| Editor | `qwen/qwen3.6-35b-a3b`, through Prime Inference |
| Sampling | Temperature 0.6; 16,384 maximum output tokens per response |
| Tools | `repair_inspect_policy`, `repair_propose_edits` |
| Patch | Complete assignments relative to baseline; at most 64, values in [−100,100] |
| Search budget | No turn, tool-call, or aggregate-output-token cap |
| Operational timeout | 60-minute rollout; HF job cap 70 minutes for setup/scoring/upload |
| Practice | 32 seeds, 4040000–4040031 |
| Stop searching | First changed candidate with practice gain >0.10 and gain >2 paired SE |
| Independent gate | 512 seeds, 4060000–4060511; same change/gain/SE rule |
| Final audit | 1,024 seeds, 4090000–4091023; never used to choose edits |
| Diagnostic miss | Seed 314159; rescue not required by this trial |
| Compute | HF Jobs CPU, Verifiers subprocess runtime |

Each unsuccessful practice proposal left the baseline unchanged. Once a proposal
qualified, the tool committed its exact diagnosis, expected behavior, and edits;
the editor could not keep searching after qualification. The trusted verifier
reproduced qualification and evaluated that frozen candidate separately.

The task supplied the game rules, current sparse weights, failure evidence, and
practice tool access. It did not supply a correct patch, a numerical candidate
solver, or a gradient training loop. The source package contains the teaching
policy generator for reproducibility; that source was not an exposed editor tool.

## Trace and selection checks

The collector checked task identity, policy fingerprint, editor model, sampling
settings, allowed tool sequence, first qualification, final arguments, submission
receipt, candidate hash, score arithmetic, seed ranges, and selected checkpoint.
The framework stopped on committed shared state before appending the final tool
response: there are 33 tool calls and 32 tool-result messages. The final receipt
and independently reproduced qualification identify the executed final proposal;
no patch was salvaged from an unexecuted request.

All 33 model requests completed without provider errors in this run. There were
29 proposal calls: 28 valid proposals and one diagnosis-length validation error.
The final valid proposal passed the independent gate.

## What the game measures

The independent 10×10 grid simulator awards +1 per brick and −1 on a paddle miss.
It samples actions from the policy's softmax, stops on a miss, clearing 24 bricks,
or 200 steps, and uses seeded randomness. Average return measures game reward,
not the fraction of episodes without failures.

The same seed is paired between baseline and candidate. Diverging actions can
produce diverging trajectories; matching seeds does not imply identical visited
states. Controlled failure replay is recorded separately from full-episode return.

Paired SE is `sample_std(candidate_return - baseline_return) / sqrt(n)`.
The gate is an engineering selection rule, not a universal safety guarantee.
Practice feedback is adaptive and is not independent statistical evidence.

## Result and provenance

The gate accepted the three-parameter edit. On the independent final audit,
mean return changed from 1.2802734375 to 4.0478515625. The gain was 2.767578125,
with paired SE 0.13622149121786994. There were 671 wins, 243 losses, and 110 ties.
The diagnostic miss was not prevented; the audit board-clear rate was zero.

The live HF Space served the same policy fingerprint and reproduced the saved
verifier episode's actions and outcome. Full model revision:
`44bef3f4fe96d48b9d67f308d7734f4693846b91`.

The earlier Prime launch `pyqeo6j8e329chclnlihsgu5` failed with “Rate exceeded”
before supplying a trace. It is retained under `results/qualified-repair-01/`
as a service failure. The completed job is
[`6aea9774c78863f9c6322027`](https://huggingface.co/jobs/HarleyCooper/6aea9774c78863f9c6322027).
Moving the task to HF compute did not change the game, weights, seeds, proposal
rule, or acceptance rule.
