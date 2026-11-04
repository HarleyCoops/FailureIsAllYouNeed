# Original PPO to self-improved: hosted two-arm pipeline

This run begins with the exact original Hugging Face PPO checkpoint, not Kimi v4
or the local experiment-09 result. The Kimi NPZ export has small rounding differences;
the launcher instead converts the pinned original PyTorch tensors without changing
their float32 values, records byte/tensor hashes, and freezes that input on HF.

| Surface | Name |
| --- | --- |
| GitHub | HarleyCoops/FailureIsAllYouNeed, codex/self-improved-pipeline |
| Original model | HarleyCooper/grid-policy-mlp |
| Base revision | 28b137f6b049eaf3d9f8fa848edb87b83f46f94a |
| Job | grid-policy-mlp-self-improved-01 |
| Container/dashboard Space | HarleyCooper/grid-policy-mlp-self-improved-training |
| Training model output | HarleyCooper/grid-policy-mlp-self-improved |
| Surgery comparison output | HarleyCooper/grid-policy-mlp-self-improved-surgery |
| Trajectory dataset | HarleyCooper/grid-policy-mlp-self-improved-trajectories |
| Inference Space | HarleyCooper/grid-policy-mlp-self-improved-demo |
| W&B | christian-cooper-us/grid-policy-mlp-self-improvement |

## Two genuinely different methods

The **training arm** uses PyTorch autograd, cross-entropy on simulator-validated
corrective paths, KL preservation against incumbent probabilities on other
observed states, and AdamW. All six trainable weight/bias tensors are optimized.
Each candidate trains for 200 optimizer steps, with preservation minibatches of
128, learning rate 0.0001 or 0.0005, and KL weights 0.1, 1, 10, 100, or 1000.
This is failure-conditioned supervised fine-tuning, not PPO from random weights.

The **surgery arm** uses the numerical W2 / W3 estimators from experiment 09.
Each arm starts from the same immutable original checkpoint and develops its own
accepted-policy history and failure queue. They share diagnosis and acceptance
rules but may encounter different failures. Equal candidate budgets do not imply
equal compute; optimizer steps, elapsed time, proposals, and checkpoints are logged.

Both arms search reachable alternative states from all original trajectory nodes,
under the exact existing physics. No action noise is injected into scored play.
Diagnostic branches are explicitly counterfactual. Both arms retain an edit only
after target replay, reset replay, retained-repair checks, and a nonregressing
96-configuration census. Incidental repairs of other starts are retained too.

## Compute and flow

The training Space builds a Docker image from a full GitHub commit SHA. Python
3.12.8 and exact Python package versions are specified. The HF Job runs this
container on cpu-basic; the dashboard and inference Space are read-only CPU apps.
No GPU or hosted editor LLM is required for this 17,253-parameter policy.

The job runs up to 50 cycles per method, 20 candidates per cycle, and stops a method
after 10 unsuccessful cycles, exhaustion of current failure cases, or no misses.
The entire job has a 90-minute timeout. A 200-step episode limit stays unchanged.
Time limits are reported separately and are not repair targets in this protocol.

After each committed cycle the job uploads losslessly compressed raw evidence,
queryable episode/transition tables, source, accepted weights, and progress to the
dataset. W&B logs actual optimizer losses, candidate metrics and the accepted-policy
curve separately. Rejected candidates cannot advance the accepted curve.

After each arm, a fresh replay checks every saved accepted checkpoint and retained
repair. Only a passed audit, a positive final mean gain, at least one accepted
checkpoint and zero regressions permit model publication. Models contain NumPy
weights, standard PyTorch-keyed safetensors, original model.py, configuration,
baseline tensors, and audit files. A release manifest pins the HF model revision.
The inference Space loads that revision and verifies its policy hash on use.

The development census is reused. It does not establish unseen generalization,
longer-horizon success, or language-model improvement. All runs, including stalled
or failed ones, remain separate. No output is called improved solely because the
optimizer loss decreased.

## Commands

Run from the repository root. The launcher uses existing local HF and W&B
credentials, passing them directly as HF Job secrets. Secrets are never written
to GitHub, the dataset, image layers, or Space variables.

```powershell
python -m unittest discover -s tests -v
python -m hosted_pipeline.launch prepare --output <launch-record-directory>
python -m hosted_pipeline.launch spaces --output <launch-record-directory> --source-commit <full-sha>
# Wait for the training Space to report RUNNING.
python -m hosted_pipeline.launch launch --output <launch-record-directory>
# After a method publishes, verify every live inference trajectory:
python -m hosted_pipeline.verify gradient --output <launch-record-directory>/deployment-gradient.json
```

The launcher records submission intent before the API call and refuses a duplicate
launch after an ambiguous response. Inspect existing HF Jobs before any retry.
An infrastructure failure preserves completed-cycle evidence. Automatic remote
resumption is not yet implemented; the local loop's receipt-based resumption
requires restored raw files and exactly matching source/runtime. Do not restart a
failed hosted run under the same output names without reconciling its state.

The live verifier checks all 96 final trajectories, the immutable model revision,
the policy fingerprint, and equality between NumPy and safetensors weights. Its
receipt is saved locally and uploaded to the trajectory dataset.

HF Job execution and Docker Space hosting follow the
[HF Jobs guide](https://huggingface.co/docs/huggingface_hub/en/guides/jobs) and
[Docker Spaces guide](https://huggingface.co/docs/hub/spaces-sdks-docker).
