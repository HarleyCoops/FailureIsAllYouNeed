"""User-authorized retrospective mean-return release; preserve original evidence."""
import argparse
import gzip
import json
from pathlib import Path

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from safetensors.numpy import save_file
from crossgame.asterix import Policy, play, serial
from crossgame.run import write

DATASET = 'HarleyCooper/asterix-recursive-trajectories-03'
REVISION = '187efee607aeccec859dd9d1d4072a016fe82321'
MODEL = 'HarleyCooper/asterix-self-improved-03'
FINGERPRINT = '70c1c9ae42992840ef20fc9352decd9a28f8eab48363e48f14d54f1d88645979'


def release(output):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    api=HfApi()
    if api.repo_exists(MODEL):raise ValueError('Release already exists; inspect before retry')
    def fetch(f):return Path(hf_hub_download(DATASET,f,repo_type='dataset',revision=REVISION))
    summary=json.loads(fetch('summary.json').read_text())
    candidate=Policy.load(fetch('final.npz'));base=Policy.load(fetch('initial.npz'))
    assert candidate.fingerprint()==FINGERPRINT
    assert summary['final_replay_passed'] and summary['held_out_audit']['gain']>0
    candidate.save(output/'policy_weights.npz');base.save(output/'initial_baseline.npz')
    write(output/'summary.json',summary)
    save_file({f'fc{i}.{kind}':np.ascontiguousarray(candidate.weights[f'W{i}'].T if kind=='weight' else candidate.weights[f'b{i}'])
               for i in [1,2,3] for kind in ['weight','bias']},str(output/'model.safetensors'))
    before=json.loads(gzip.decompress(fetch('audit-base.json.gz').read_bytes()))
    after=json.loads(gzip.decompress(fetch('audit-final.json.gz').read_bytes()))
    diffs=np.array([b['total']-a['total'] for a,b in zip(before,after)])
    rng=np.random.default_rng(20261105)
    interval=np.quantile(rng.choice(diffs,size=(10000,len(diffs)),replace=True).mean(axis=1),[.025,.975]).tolist()
    decision=dict(published=True,criterion='Positive observed fresh-audit mean gain; individual regressions allowed',
                  retrospective_rule_change=True,original_strict_gate_passed=False,
                  authorization='User explicitly requested removing individual-game preservation veto and publishing run03',
                  evidence_dataset=DATASET,evidence_revision=REVISION,policy_hash=FINGERPRINT,
                  paired_mean_bootstrap_95_interval=interval,bootstrap_replicates=10000,bootstrap_seed=20261105)
    write(output/'release-decision.json',decision)
    # Replay one improved and one regressed audit game, selected by first seed in each class.
    chosen=[next(a['seed'] for a,b in zip(before,after) if b['total']>a['total']),
            next(a['seed'] for a,b in zip(before,after) if b['total']<a['total'])]
    checks=[]
    for seed in chosen:
        for variant,policy,rows in [('base',base,before),('improved',candidate,after)]:
            ep=play(policy,seed,1000,trace=True)
            expected=next(r for r in rows if r['seed']==seed)
            assert {k:v for k,v in serial(ep).items() if k!='worlds'}==expected
            write(output/'simulation'/f'{seed}-{variant}.json.gz',serial(ep))
            checks.append(dict(seed=seed,variant=variant,total=ep['total'],steps=ep['steps'],reason=ep['reason']))
    write(output/'simulation-checks.json',checks)
    card=f'''---
license: cc-by-4.0
tags:
- reinforcement-learning
- minatar
- asterix
- recursive-learning
---
# Asterix self-improved 03: measured average-return improvement

This is a 20,486-parameter greedy Asterix policy improved through failure-conditioned
gradient updates and recursive regression memory. It continues from run 02's research
checkpoint. Original weights were authored by Astra in Rikiya Takehi and collaborators'
`RikiyaT/lm_understand-logs`; this is not an original PPO policy.

## What improved, and what did not

| Evaluation | Starting run 02 mean | This model mean | Wins / losses / ties |
| --- | ---: | ---: | --- |
| Development, 160 games | 57.76875 | 59.0875 | 27 / 0 / 133 |
| Selection, 32 games | 62.03125 | 62.0625 | 1 / 0 / 31 |
| Fresh audit, 128 games | 56.0078125 | 56.4375 | 14 / 10 / 104 |

The audit gain is **+0.4296875 mean return**, approximately 0.77%. There are **10
lower-scoring games and five lost survivors** versus run 02. A lost survivor is a
game that previously reached the 1,000-step limit but now ends in a collision.
These categories overlap. The paired bootstrap 95% interval for mean gain is
**[{interval[0]:.4f}, {interval[1]:.4f}]**, using 10,000 resamples of these 128 seed
pairs. This descriptive interval does not correct for retrospective rule selection
or establish superiority on all games.

Against the original policy on the same audit, mean return is 55.8671875 → 56.4375,
with 32 wins, 26 losses, 70 ties, and nine lost survivors. On reused training games,
the previous original-reference losses fell from 26 to 9 and lost survivors from
11 to 6. Reused games are development evidence, not independent transfer evidence.

## Explicit release-rule change

**This checkpoint failed the original zero-regression publication gate.** After
reviewing the results, the user explicitly chose an average-return objective and
authorized this release. The original `summary.json` is preserved and still says
`publishable: false` under that original rule. `release-decision.json` records the
separate retrospective release decision. The model name denotes measured average
improvement, not universal preservation or statistically established superiority.

## How the weights changed

Run 03 made 67 bounded gradient-training proposals across six cycles and accepted
one checkpoint under its original strict development checks. Training combined
verified legal corrective trajectories, preservation probes, and a regression memory
that grew to 618 states. All four inherited repairs remained valid in final replay.
The accepted network changes actions through its weights alone: no runtime planner,
external repair system, or changed game physics is needed for inference.

The network is 600 → 32 → 32 → 6 with tanh hidden layers. `policy_weights.npz`
contains W1/b1, W2/b2, W3/b3 in input-by-output matrix orientation;
`model.safetensors` contains equivalent fc1/fc2/fc3 tensors with transposed linear
weight matrices. The observation adapter and native runtime are required; this is
not a Transformers text model. `initial_baseline.npz` is run 02, not the original.

## Reproduce and inspect

Use the pinned source's `crossgame.asterix.Policy.load` and `play(policy, seed, 1000)`.
The game is official MinAtar 1.0.15 with native 0.1 sticky actions and difficulty
ramping. Policy actions are greedy. Results are not comparable to an upstream
10,000-step sampled-action score without matching that evaluation protocol.

- [Frozen source](https://github.com/HarleyCoops/FailureIsAllYouNeed/tree/5597b6758f20ada54686cc250c86fae5b00af361)
- [Immutable experiment evidence](https://huggingface.co/datasets/{DATASET}/tree/{REVISION})
- [Paired simulator](https://huggingface.co/spaces/HarleyCooper/asterix-recursive-learning-03)
- [W&B curves](https://wandb.ai/christian-cooper-us/failure-is-all-you-need-crossgame/runs/5c7f3a56)

Policy fingerprint: `{FINGERPRINT}`. The release includes independent exact replays
of one improved and one regressed audit game. The next experiment starts from these
weights, moves the former audit into training, and reserves a new 512-game audit.

Original policy weights and upstream material: Rikiya Takehi and collaborators,
CC BY 4.0. MinAtar game: Kenny Young and Tian Tian, GPL-3.0. See the pinned source's
licenses and provenance; the game engine is not bundled into these weight tensors.
'''
    (output/'README.md').write_text(card,encoding='utf-8')
    api.create_repo(MODEL,private=False)
    commit=api.upload_folder(repo_id=MODEL,folder_path=output,commit_message='Release run03 under user-authorized average-return criterion')
    downloaded=Policy.load(hf_hub_download(MODEL,'policy_weights.npz',revision=commit.oid))
    assert downloaded.fingerprint()==FINGERPRINT
    latest=json.loads(fetch('latest.json').read_text())
    latest['releases']['improved']=dict(repo=MODEL,revision=commit.oid,policy_hash=FINGERPRINT)
    latest['release_decision']=decision
    write(output/'published-latest.json',latest)
    api.upload_file(repo_id=DATASET,repo_type='dataset',path_or_fileobj=output/'published-latest.json',path_in_repo='latest.json',commit_message='Link authorized average-return release; preserve original strict audit result')
    receipt=dict(repo=MODEL,revision=commit.oid,policy_hash=FINGERPRINT,checks=checks,bootstrap_interval=interval)
    write(output/'publication-receipt.json',receipt)
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);release(p.parse_args().output)
