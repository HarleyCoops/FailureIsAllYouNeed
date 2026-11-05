"""Official MinAtar adapter, exact RNG-preserving replay, and failure diagnosis."""
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from minatar import Environment

KEYS = {'W1': 'actor/dense1/kernel', 'b1': 'actor/dense1/bias',
        'W2': 'actor/dense2/kernel', 'b2': 'actor/dense2/bias',
        'W3': 'actor/logits/kernel', 'b3': 'actor/logits/bias'}
SHAPES = {'W1': (600,32), 'b1': (32,), 'W2': (32,32), 'b2': (32,), 'W3': (32,6), 'b3': (6,)}


class Policy:
    def __init__(self, weights):
        self.weights = {k: np.asarray(weights[k], dtype=np.float32).copy() for k in SHAPES}
        for k, shape in SHAPES.items():
            if self.weights[k].shape != shape or not np.isfinite(self.weights[k]).all():
                raise ValueError('Invalid policy tensor: ' + k)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as z:
            return cls({k: z[k] if k in z else z[v] for k,v in KEYS.items()})

    def save(self, path):
        np.savez_compressed(path, **self.weights)

    def logits(self, obs):
        w = self.weights
        return np.tanh(np.tanh(obs @ w['W1'] + w['b1']) @ w['W2'] + w['b2']) @ w['W3'] + w['b3']

    def act(self, obs):
        return int(self.logits(obs).argmax())

    def fingerprint(self):
        h = hashlib.sha256()
        for k,v in sorted(self.weights.items()):
            h.update(k.encode()); h.update(v.astype('<f4').tobytes())
        return h.hexdigest()


class Asterix:
    """Policy is greedy; official environment retains its native 0.1 sticky actions."""
    def __init__(self, seed, horizon=1000):
        self.game = Environment('asterix', sticky_action_prob=0.1, difficulty_ramping=True)
        self.game.seed(seed); self.game.reset()
        self.seed, self.horizon = seed, horizon
        self.steps, self.total, self.dead = 0, 0., False

    @property
    def done(self):
        return self.dead or self.steps >= self.horizon

    def observation(self):
        raw = self.game.state()
        obs = np.zeros((10,10,6), dtype=np.float32)
        obs[:,:,0] = raw[:,:,0]; obs[:,:,1] = raw[:,:,1]
        obs[:,:,2] = raw[:,:,3]; obs[:,:,4] = raw[:,:,2]
        return obs.reshape(600)

    def step(self, action):
        if self.done:
            raise ValueError('Episode ended')
        reward, self.dead = self.game.act(int(action))
        self.steps += 1; self.total += float(reward)

    def snapshot(self):
        raw = self.game.env
        rng = self.game.random.get_state()
        # One RNG object is shared by native sticky actions and entity spawning.
        return dict(step=self.steps, total=self.total, dead=bool(self.dead),
                    player=[raw.player_x, raw.player_y], entities=copy.deepcopy(raw.entities),
                    last_action=self.game.last_action,
                    timers={k: getattr(raw,k) for k in ('spawn_timer','move_timer','ramp_timer','ramp_index','spawn_speed','move_speed')},
                    rng=dict(kind=rng[0], keys=rng[1].tolist(), pos=int(rng[2]), has_gauss=int(rng[3]), cached=float(rng[4])))


def play(policy, seed, horizon=1000, prefix=(), trace=False):
    env = Asterix(seed,horizon)
    actions, observations, snapshots, worlds = [], [], [], []
    while not env.done:
        obs = env.observation()
        if trace:
            observations.append(obs); snapshots.append(copy.deepcopy(env)); worlds.append(env.snapshot())
        action = prefix[env.steps] if env.steps < len(prefix) else policy.act(obs)
        actions.append(int(action)); env.step(action)
    result = dict(seed=seed, total=env.total, steps=env.steps,
                  reason='collision' if env.dead else 'time_limit', actions=actions, final=env.snapshot())
    if trace:
        result.update(observations=np.asarray(observations), snapshots=snapshots, worlds=worlds)
    return result


def serial(ep):
    return {k:v for k,v in ep.items() if k not in ('observations','snapshots')}


def evaluate(policy, seeds, horizon):
    return [play(policy,int(seed),horizon) for seed in seeds]


def compare(before, after):
    assert [x['seed'] for x in before] == [x['seed'] for x in after]
    differences = np.array([a['total']-b['total'] for b,a in zip(before,after)])
    return dict(before_mean=float(np.mean([x['total'] for x in before])),
                after_mean=float(np.mean([x['total'] for x in after])),
                gain=float(differences.mean()), wins=int((differences>0).sum()),
                losses=int((differences<0).sum()), ties=int((differences==0).sum()),
                lost_survivors=sum(b['reason']=='time_limit' and a['reason']=='collision' for b,a in zip(before,after)))


def diagnose(policy, episode, *, max_anchors=24, depth=4, width=6, rollout_budget=96):
    """Multi-action branch search across stratified full-trajectory nodes.

    Clones retain exact RNG and sticky-action history. Search cannot inject a
    world state or change a random seed. Bounded coverage is saved explicitly.
    """
    n = len(episode['actions'])
    anchors = sorted(set(np.linspace(0,n-1,min(n,max_anchors),dtype=int).tolist()))
    budget_per_anchor = max(1, rollout_budget // len(anchors))
    targets, evidence = [], []
    for anchor in anchors:
        frontier = [(copy.deepcopy(episode['snapshots'][anchor]), [], [])]
        anchor_rollouts = 0
        for level in range(depth):
            branches = []
            for env, actions, observations in frontier:
                for action in range(5):
                    branch = copy.deepcopy(env)
                    obs = branch.observation(); branch.step(action)
                    if branch.dead:
                        continue
                    sequence, inputs = actions+[action], observations+[obs]
                    branches.append((branch,sequence,inputs))
            # Rank only diagnostic branches, never alter actual inference.
            branches.sort(key=lambda x:(-x[0].total, sum(a != policy.act(o) for a,o in zip(x[1],x[2])),x[1]))
            frontier = [x for x in branches[:width] if not x[0].done]
            if level == depth-1 or not frontier:
                for branch, actions, observations in branches:
                    if anchor_rollouts >= budget_per_anchor:
                        break
                    if actions == episode['actions'][anchor:anchor+len(actions)]:
                        continue
                    trial = copy.deepcopy(branch)
                    while not trial.done:
                        trial.step(policy.act(trial.observation()))
                    anchor_rollouts += 1
                    good = trial.total > episode['total'] and trial.steps > episode['steps']
                    evidence.append(dict(anchor=anchor, sequence=actions, total=trial.total,
                                         steps=trial.steps, dead=bool(trial.dead), corrected=good))
                    if good:
                        targets.append(dict(anchor=anchor, actions=actions, observations=np.asarray(observations),
                                            total=trial.total, steps=trial.steps))
            if not frontier:
                break
    targets.sort(key=lambda t:(-t['total'],-t['steps'],t['anchor']))
    return targets[:2], dict(trajectory_nodes=n, anchors=anchors, depth=depth,
                             width=width, rollouts=len(evidence), trials=evidence)


def train(policy, targets, probes, *, lr=0.0001, penalty=1., steps=300, seed=9042026):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1); torch.manual_seed(seed); torch.use_deterministic_algorithms(True)
    network = nn.Sequential(nn.Linear(600,32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,6))
    with torch.no_grad():
        for i,layer in enumerate((network[0],network[2],network[4]),1):
            layer.weight.copy_(torch.from_numpy(policy.weights[f'W{i}'].T.copy()))
            layer.bias.copy_(torch.from_numpy(policy.weights[f'b{i}'].copy()))
    start = [p.detach().clone() for p in network.parameters()]
    x = np.concatenate([t['observations'] for t in targets])
    y = np.concatenate([t['actions'] for t in targets])
    target_bytes = {o.tobytes() for o in x}
    probes = np.asarray([o for o in probes if o.tobytes() not in target_bytes])
    if not len(probes):
        raise ValueError('No preservation observations')
    x,y,probes = torch.from_numpy(x),torch.tensor(y,dtype=torch.long),torch.from_numpy(probes)
    with torch.no_grad():
        teacher = torch.softmax(network(probes),-1)
    optimizer = torch.optim.AdamW(network.parameters(),lr=lr,weight_decay=0.)
    history=[]
    for step in range(steps):
        indices = torch.randint(len(probes),(min(128,len(probes)),))
        optimizer.zero_grad()
        ce = F.cross_entropy(network(x),y)
        kl = F.kl_div(F.log_softmax(network(probes[indices]),-1),teacher[indices],reduction='batchmean')
        displacement = sum((p-q).square().sum() for p,q in zip(network.parameters(),start))
        loss = ce + penalty*kl + 1e-4*displacement
        loss.backward(); torch.nn.utils.clip_grad_norm_(network.parameters(),1.); optimizer.step()
        if step==0 or (step+1)%50==0:
            history.append(dict(step=step+1,loss=float(loss.detach()),ce=float(ce.detach()),kl=float(kl.detach())))
    weights={}
    for i,layer in enumerate((network[0],network[2],network[4]),1):
        weights[f'W{i}']=layer.weight.detach().numpy().T.copy()
        weights[f'b{i}']=layer.bias.detach().numpy().copy()
    return Policy(weights),history
