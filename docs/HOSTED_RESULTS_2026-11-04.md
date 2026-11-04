# Hosted original-base self-improvement — November 4, 2026

The single HF Job completed both experiments successfully. Both started from the
exact original PPO tensors, not Kimi v4 or a repaired checkpoint. Rules, physics,
greedy action selection, and the 200-step horizon stayed fixed.

| Method | Base mean | Final mean | Accepted updates | Improved / regressed / tied starts | Final misses |
| --- | ---: | ---: | ---: | --- | ---: |
| Failure-conditioned AdamW training | 8.895833 | 16.750000 | 6 | 56 / 0 / 40 | 6 |
| Direct parameter surgery | 8.895833 | 8.895833 | 0 | 0 / 0 / 96 | 56 |

The training arm used actual PyTorch backpropagation over all policy layers,
cross-entropy on valid corrective paths, and KL preservation against the incumbent
on other observed states. It evaluated 226 candidate trainings over 14 cycles,
totaling 45,200 optimizer steps across separate candidates. Rejected candidates
did not replace the accepted model. The final bank retains 74 correction records
covering 56 distinct starting configurations. The arm stopped after attempting
all remaining current failure cases within its registered diagnostic budget.

Surgery evaluated 170 candidates in 10 cycles and stopped without acceptance.
Its unchanged base was retained; no model was published under the planned
`grid-policy-mlp-self-improved-surgery` name. The successful earlier local run
started from v4 and remains a different experiment.

## Published surfaces

- [Completed HF Job](https://huggingface.co/jobs/HarleyCooper/6aebbce80e0338d1bb804606)
- [Training container and dashboard](https://huggingface.co/spaces/HarleyCooper/grid-policy-mlp-self-improved-training)
- [Trained self-improved model](https://huggingface.co/HarleyCooper/grid-policy-mlp-self-improved/tree/49d08edd1d60af4a8b82dfabb74b4c8edc7c3187)
- [Live inference Space](https://huggingface.co/spaces/HarleyCooper/grid-policy-mlp-self-improved-demo)
- [Trajectory and experiment dataset](https://huggingface.co/datasets/HarleyCooper/grid-policy-mlp-self-improved-trajectories)
- [Consolidated W&B report and curves](https://wandb.ai/christian-cooper-us/grid-policy-mlp-self-improvement/reports/Original-PPO-self-improvement---gradient-training-and-parameter-surgery--VmlldzoxODI1NTgwNA==)
- [Frozen executable source](https://github.com/HarleyCoops/FailureIsAllYouNeed/tree/e7845f15f172344d918c9d8085898fad2a9e8768)
- [Pipeline configuration and names](HOSTED_PIPELINE.md)

## Verification and scope

All 13 prelaunch software checks passed. Both final checkpoint histories passed
the full replay audit. W&B summaries matched the saved results. All 96 final
gradient-model episodes served by the live inference Space matched the saved
trajectories, including every action and world state, rewards and termination.
The published safetensors and NumPy exports have identical tensors.
The [deployment receipt](https://huggingface.co/datasets/HarleyCooper/grid-policy-mlp-self-improved-trajectories/blob/main/runs/self-improved-20261104-01/deployment-gradient.json)
pins the model revision, dataset revision, and fingerprint.

The final trained policy had **six misses, 90 time limits, and zero board clears**.
The reported preservation applies to the exact reused 96-configuration development
population and horizon, not unseen situations or longer episodes. Equal return
does not imply unchanged actions. Neither this run nor the earlier local repair
demonstrates language-model improvement or a general ranking of the two methods.

The input model fingerprint is
`c818ab03938861603c446365e414296c1e250667e9814f85168d166fd714c6cf`;
the trained fingerprint is
`ea45539e3b1d58c5eae723ead9e9204c13a53900e762672949a3cc4b0c6c9e3d`.
