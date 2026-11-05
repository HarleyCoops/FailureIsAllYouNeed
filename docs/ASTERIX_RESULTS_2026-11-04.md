# Asterix cross-game trial: completed

The hosted job completed six cycles and 48 candidate trainings. **No checkpoint was accepted; no improved model was published.**

- 41 candidates repaired their target under prefix-conditioned and full-reset replay, but failed preservation.
- Seven candidates did not repair the target.
- 17 candidates improved both development and selection-gate means, while losing performance or survival on individual seeds.
- The original policy remains the inference baseline. Its fingerprint is unchanged.

| Split | Episodes | Base and final accepted mean |
| --- | ---: | ---: |
| Development | 32 | 53.40625 |
| Reused selection gate | 32 | 59.4375 |
| Reserved final audit | 128 | 56.984375 |

The final audit ran after selection and evaluated the unchanged incumbent. It does not validate the rejected candidates. Native MinAtar rules, difficulty ramping and sticky actions were retained. This 1,000-step greedy-policy protocol differs from upstream sampled-action 10,000-step scoring.

The base was the Astra-authored Asterix network from RikiyaT/lm_understand-logs, not PPO. Five tests, an end-to-end smoke run, live baseline inference verification, and final hosted replay passed.

The method can diagnose and train target repairs in this new game, but **accepted recursive improvement has not yet been demonstrated here**. The next research direction is to include the newly exposed regressions as joint training constraints, under a new protocol and fresh final audit. The completed run's gate is not changed retroactively.

[W&B report](https://wandb.ai/christian-cooper-us/failure-is-all-you-need-crossgame/reports/Asterix-recursive-learning---first-cross-game-trial--VmlldzoxODI1NjA0Nw==) · [HF Job](https://huggingface.co/jobs/HarleyCooper/6aebce4ac78863f9c6330f73) · [Playable simulator](https://huggingface.co/spaces/HarleyCooper/asterix-recursive-learning-01) · [All evidence](https://huggingface.co/datasets/HarleyCooper/asterix-recursive-trajectories-01) · [Frozen protocol](https://github.com/HarleyCoops/FailureIsAllYouNeed/blob/3c681ad8486ffb82d4365321c8669193a67bcc67/docs/ASTERIX_RECURSIVE.md)
