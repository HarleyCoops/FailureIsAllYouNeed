"""Actual failure-conditioned supervised training with AdamW and backpropagation.

Correction labels come from legal counterfactual paths. Preservation uses KL to
the incumbent on other observed states. Exact episode gates remain authoritative.
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from experiment09.loop import Policy, policy_hash


class TrainablePolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.fc1, self.fc2, self.fc3 = nn.Linear(500, 32), nn.Linear(32, 32), nn.Linear(32, 5)
        with torch.no_grad():
            for i in (1, 2, 3):
                layer = getattr(self, f"fc{i}")
                layer.weight.copy_(torch.from_numpy(policy.weights[f"W{i}"].T.copy()))
                layer.bias.copy_(torch.from_numpy(policy.weights[f"b{i}"].copy()))

    def forward(self, x):
        return self.fc3(torch.tanh(self.fc2(torch.tanh(self.fc1(x)))))

    def to_policy(self):
        weights = {}
        for i in (1, 2, 3):
            layer = getattr(self, f"fc{i}")
            weights[f"W{i}"] = layer.weight.detach().cpu().numpy().T.copy()
            weights[f"b{i}"] = layer.bias.detach().cpu().numpy().copy()
        return Policy(weights)


def train_candidate(policy, target, probes, layer, penalty, margin=0.1, cap=2.0,
                    *, steps=200, seed=9042026, batch_size=128):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    network = TrainablePolicy(policy)
    initial = [p.detach().clone() for p in network.parameters()]
    x = np.asarray(target.get("sequence_observations", [target["observation"]]), dtype=np.float32)
    y = np.asarray(target.get("sequence_actions", [target["action"]]), dtype=np.int64)
    correction_x, correction_y = torch.from_numpy(x), torch.from_numpy(y)
    # Do not simultaneously demand old and corrective labels at the same observation.
    target_bytes = {obs.tobytes() for obs in x}
    retained = np.asarray([o for o in probes if o.tobytes() not in target_bytes], dtype=np.float32)
    if not len(retained):
        return None, dict(reason="no_preservation_examples")
    preservation_x = torch.from_numpy(retained)
    with torch.no_grad():
        teacher = torch.softmax(network(preservation_x), dim=-1)
    learning_rate = 1e-4 if layer == "adam-low" else 5e-4
    optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed)
    history = []
    for step in range(steps):
        indices = torch.randint(len(retained), (min(batch_size, len(retained)),), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        correction_loss = F.cross_entropy(network(correction_x), correction_y)
        preserve_loss = F.kl_div(F.log_softmax(network(preservation_x[indices]), dim=-1),
                                teacher[indices], reduction="batchmean")
        displacement = sum((p-q).square().sum() for p, q in zip(network.parameters(), initial))
        loss = correction_loss + penalty * preserve_loss + 1e-4 * displacement
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(network.parameters(), 1.0)
        optimizer.step()
        if step == 0 or (step+1) % 20 == 0 or step == steps-1:
            history.append(dict(optimizer_step=step+1, loss=float(loss.detach()),
                                correction_ce=float(correction_loss.detach()),
                                preservation_kl=float(preserve_loss.detach()),
                                gradient_norm=float(norm)))
    candidate = network.to_policy()
    delta_norm = float(np.sqrt(sum(np.sum((candidate.weights[k].astype(float)-v)**2)
                                  for k, v in policy.weights.items())))
    with torch.no_grad():
        accuracy = float((network(correction_x).argmax(-1) == correction_y).float().mean())
    return candidate, dict(method="gradient", optimizer="AdamW", learning_rate=learning_rate,
                           steps=steps, penalty=penalty, delta_norm=delta_norm,
                           training_accuracy=accuracy, history=history,
                           changed=policy_hash(candidate) != policy_hash(policy),
                           parameter_tensors_trained=[n for n, _ in network.named_parameters()])
