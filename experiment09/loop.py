"""Bounded numerical repair pilot. Run with python -m experiment09.loop.

Only diagnostic counterfactuals override actions. All scored policy episodes use
the vendored float32 Policy.act(..., 'greedy') and unmodified Breakout engine.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor"))
from weight_space_breakout.game import Breakout
from weight_space_breakout.policy import Policy


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def policy_hash(policy):
    h = hashlib.sha256()
    for key, value in sorted(policy.weights.items()):
        h.update(key.encode())
        h.update(np.asarray(value, dtype="<f4").tobytes())
    return h.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_policy(path, policy):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    with temp.open("wb") as f:
        np.savez_compressed(f, **policy.weights)
    os.replace(temp, path)


def state(game):
    return dict(step=game.steps, paddle=game.paddle, row=game.row, col=game.col,
                dx=game.dx, dy=game.dy, bricks=sorted(map(list, game.bricks)),
                previous=np.flatnonzero(game.previous).tolist(), total=game.total,
                reason=game.reason, terminated=game.terminated, truncated=game.truncated)


def witnesses():
    expected = set(itertools.product(range(2, 8), range(1, 9), (-1, 1)))
    found = {}
    game = Breakout()
    for seed in range(10000):
        game.reset(seed)
        config = game.paddle, game.col, game.dx
        if config not in expected or game.row != 6 or game.dy != -1:
            raise ValueError("Reset support changed")
        if len(game.bricks) != 24 or game.previous.any():
            raise ValueError("Initial state changed")
        found.setdefault(config, seed)
        if set(found) == expected:
            return [dict(config=list(c), seed=found[c]) for c in sorted(found)]
    raise ValueError("Could not verify all 96 legal configurations")


def play(policy, seed, *, prefix=(), override=None, trace=False):
    game = Breakout(max_steps=200)
    game.reset(seed)
    actions, states, observations = [], [state(game)], []
    while not (game.terminated or game.truncated):
        obs = game.observation()
        t = game.steps
        if t < len(prefix):
            action = prefix[t]
        elif override is not None and t == override[0]:
            action = override[1]
        else:
            action = policy.act(obs, None, "greedy")
        if trace:
            observations.append(obs)
        actions.append(action)
        game.step(action)
        states.append(state(game))
    result = dict(seed=seed, total=game.total, reason=game.reason, steps=game.steps,
                  actions=actions, states=states)
    if trace:
        result["observations"] = np.asarray(observations, dtype=np.float32)
    return result


def census(policy, cases, trace=False):
    return [play(policy, c["seed"], trace=trace) for c in cases]


def serial_episode(episode):
    return {k: v for k, v in episode.items() if k != "observations"}


def case_from_episode(episode, checkpoint):
    # Freeze through the last successful paddle contact, before the final miss.
    contacts = [s["step"] for s in episode["states"][1:-1]
                if s["row"] == 8 and s["dy"] == -1 and not s["terminated"]]
    start = max(contacts, default=0)
    identity = dict(checkpoint=checkpoint, seed=episode["seed"],
                    actions=episode["actions"], final=episode["states"][-1])
    identity_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return dict(id=identity_hash, seed=episode["seed"], checkpoint=checkpoint,
                prefix=episode["actions"][:start], target_step=episode["steps"],
                target_col=episode["states"][-1]["col"], baseline_return=episode["total"],
                episode=serial_episode(episode))


def rescued(episode, failure):
    if failure.get("world_target"):
        return world_corrected(episode, failure)
    step = failure["target_step"]
    if len(episode["states"]) <= step:
        return False
    s = episode["states"][step]
    return (s["row"] == 8 and s["col"] == failure["target_col"]
            and s["dy"] == -1 and s["reason"] != "miss")


def world_corrected(episode, failure):
    """A different physical path may avoid the failure; no same-ball constraint."""
    return (episode["total"] > failure["baseline_return"] and
            (episode["reason"] == "cleared" or episode["steps"] > failure["target_step"]))


def diagnose(policy, episode, failure, lookback=12):
    """Single-action interventions in isolated replay; never discovery noise."""
    outcomes, targets = [], []
    start = max(len(failure["prefix"]), episode["steps"] - lookback)
    for t in range(episode["steps"] - 1, start - 1, -1):
        for action in (0, 3, 4):  # up/down are exact no-op equivalents in this engine
            if action == episode["actions"][t]:
                continue
            trial = play(policy, episode["seed"], prefix=episode["actions"][:t],
                         override=(t, action))
            fixed = rescued(trial, failure)
            record = dict(step=t, action=action, rescued=fixed, total=trial["total"])
            outcomes.append(record)
            if fixed and trial["total"] > episode["total"]:
                targets.append(dict(**record, observation=episode["observations"][t]))
    # Prefer strong full-episode counterfactuals, then earlier interventions.
    targets.sort(key=lambda x: (-x["total"], x["step"], x["action"]))
    return targets[:2], outcomes


def diagnose_sequence(policy, episode, failure, lookback=12, width=64):
    """Search legal action sequences in cloned games, never alter physics.

    Merge identical complete world states, retaining the path with fewest
    deviations from incumbent decisions. A bounded frontier is a diagnostic
    search, not an assertion of global feasibility or an exploration policy.
    """
    start = max(len(failure["prefix"]), episode["steps"] - lookback)
    game = Breakout()
    game.reset(episode["seed"])
    for action in episode["actions"][:start]:
        game.step(action)
    frontier = [(game, [], [], 0)]
    stats = []
    for t in range(start, episode["steps"]):
        merged = {}
        expansions = 0
        for current, actions, observations, deviations in frontier:
            obs = current.observation()
            greedy = policy.act(obs, None, "greedy")
            for action in sorted({greedy, 0, 3, 4}):
                branch = copy.deepcopy(current)
                branch.step(action)
                expansions += 1
                if branch.reason == "miss":
                    continue
                cost = deviations + int(action != greedy)
                candidate = (branch, actions + [action], observations + [obs], cost)
                key = json.dumps(state(branch), sort_keys=True)
                if key not in merged or cost < merged[key][3]:
                    merged[key] = candidate
        frontier = sorted(merged.values(), key=lambda x: (x[3], abs(x[0].paddle-failure["target_col"]), x[1]))[:width]
        stats.append(dict(step=t, expansions=expansions, unique_states=len(merged), retained=len(frontier)))
        if not frontier:
            break
    targets = []
    for game, actions, observations, cost in frontier:
        if game.steps != failure["target_step"] or game.row != 8 or game.col != failure["target_col"] or game.dy != -1:
            continue
        full_prefix = episode["actions"][:start] + actions
        trial = play(policy, episode["seed"], prefix=full_prefix)
        if trial["total"] <= episode["total"]:
            continue
        first_change = next((i for i, (obs, a) in enumerate(zip(observations, actions))
                             if policy.act(obs, None, "greedy") != a), None)
        if first_change is None:
            continue
        targets.append(dict(step=start+first_change, action=actions[first_change],
                            observation=observations[first_change], rescued=True,
                            total=trial["total"], deviations=cost,
                            sequence_start=start, sequence_actions=actions,
                            sequence_observations=np.asarray(observations)))
    targets.sort(key=lambda x: (x["deviations"], -x["total"], x["sequence_actions"]))
    return targets[:2], stats


def finish_world(policy, game):
    game = copy.deepcopy(game)
    while not (game.terminated or game.truncated):
        game.step(policy.act(game.observation(), None, "greedy"))
    return dict(total=game.total, steps=game.steps, reason=game.reason)


def diagnose_worlds(policy, episode, failure, other_episodes=(), *, depth_limit=6,
                    state_budget=2048, rollout_budget=512):
    """Search reachable alternative world states across the WHOLE trajectory.

    Breadth-first across every original decision point prevents a last-N-steps
    restriction. Paths are legal transitions, never fabricated observations.
    Budgets bound coverage; this is not exhaustive search over every possible path.
    """
    support = {}
    for ep in other_episodes:
        for obs in {o.tobytes() for o in ep["observations"]}:
            support[obs] = support.get(obs, 0) + 1
    game = Breakout()
    game.reset(episode["seed"])
    frontier = []
    for anchor, action in enumerate(episode["actions"]):
        frontier.append((copy.deepcopy(game), anchor, [], [], 0, 0.0))
        game.step(action)
    seen, targets, evidence = set(), [], []
    rollouts = expansions = 0
    anchors_examined = set()
    for depth in range(1, depth_limit + 1):
        next_frontier = []
        for current, anchor, actions, observations, deviations, gap_cost in frontier:
            if current.terminated or current.truncated:
                continue
            obs = current.observation()
            logits = policy.logits(obs)
            greedy = policy.act(obs, None, "greedy")
            for action in sorted({greedy, 0, 3, 4}):
                if depth == 1 and (action == greedy or action < 3 and greedy < 3):
                    continue
                if expansions >= state_budget:
                    break
                branch = copy.deepcopy(current)
                branch.step(action)
                expansions += 1
                anchors_examined.add(anchor)
                key = json.dumps(state(branch), sort_keys=True)
                if key in seen or branch.reason == "miss":
                    continue
                seen.add(key)
                new_actions, new_obs = actions + [action], observations + [obs]
                cost = deviations + int(action != greedy)
                gaps = gap_cost + max(0., float(logits.max()-logits[action]))
                path = (branch, anchor, new_actions, new_obs, cost, gaps)
                next_frontier.append(path)
                if rollouts >= rollout_budget:
                    continue
                result = finish_world(policy, branch)
                rollouts += 1
                corrected = world_corrected(result, failure)
                candidate_id = hashlib.sha256(key.encode()).hexdigest()
                evidence.append(dict(world_id=candidate_id, anchor=anchor,
                                     depth=depth, actions=new_actions, reachable_world=state(branch),
                                     continuation=result, corrected=corrected,
                                     action_deviations=cost, logit_gap_cost=gaps))
                if corrected:
                    shared = max((support.get(o.tobytes(), 0) for o in new_obs), default=0)
                    targets.append(dict(step=anchor, action=new_actions[0], observation=new_obs[0],
                                        rescued=True, total=result["total"], deviations=cost,
                                        logit_gap_cost=gaps, shared_observation_cases=shared,
                                        sequence_start=anchor, sequence_actions=new_actions,
                                        sequence_observations=np.asarray(new_obs),
                                        desired_world=state(branch), world_id=candidate_id))
            if expansions >= state_budget:
                break
        if not next_frontier or expansions >= state_budget or rollouts >= rollout_budget:
            break
        frontier = sorted(next_frontier, key=lambda x: (x[4], x[5], x[1], x[2]))
    targets.sort(key=lambda x: (x["deviations"], -x["shared_observation_cases"],
                                x["logit_gap_cost"], -x["total"], x["step"]))
    # Distinct anchors give two genuinely different causal hypotheses when possible.
    selected = []
    for target in targets:
        if not selected or target["step"] != selected[0]["step"]:
            selected.append(target)
        if len(selected) == 2:
            break
    if len(selected) == 1 and len(targets) > 1:
        selected.append(next(t for t in targets if t is not selected[0]))
    stats = dict(anchors_total=episode["steps"], anchors_examined=sorted(anchors_examined),
                 expansions=expansions, continuation_rollouts=rollouts,
                 successful_worlds=len(targets), budget_exhausted=(expansions >= state_budget or rollouts >= rollout_budget),
                 reachable_worlds=evidence)
    return selected, stats


def activations(policy, observations):
    w = policy.weights
    h1 = np.tanh(observations @ w["W1"] + w["b1"])
    h2 = np.tanh(h1 @ w["W2"] + w["b2"])
    return h1.astype(np.float64), h2.astype(np.float64)


def margin_system(policy, observation, action, layer):
    h1, h2 = activations(policy, observation)
    logits = policy.logits(observation).astype(np.float64)
    competitors = [a for a in range(5) if a != action]
    rows = []
    for other in competitors:
        if layer == "W3":
            grad = np.zeros((33, 5))
            grad[:, action] = np.r_[h2, 1.0]
            grad[:, other] = -np.r_[h2, 1.0]
        else:
            d = (1 - h2 ** 2) * (policy.weights["W3"][:, action]
                                 - policy.weights["W3"][:, other])
            grad = h1[:, None] * d[None, :]
        rows.append(grad.ravel())
    return np.asarray(rows), logits[action] - logits[competitors]


def target_system(policy, target, layer):
    observations = target.get("sequence_observations", [target["observation"]])
    actions = target.get("sequence_actions", [target["action"]])
    systems = [margin_system(policy, obs, action, layer) for obs, action in zip(observations, actions)]
    return np.concatenate([a for a, _ in systems]), np.concatenate([b for _, b in systems])


def dual_projection(A, required, inverse_apply, sweeps=2000):
    """Minimize delta' M delta / 2 subject to A delta >= required.

    Coordinate ascent on the nonnegative dual handles all competing actions,
    rather than only the previously winning action. Infeasibility is reported
    as nonconvergence, not a proof that no policy repair exists.
    """
    directions = np.stack([inverse_apply(row) for row in A])
    gram = A @ directions.T
    alpha = np.zeros(len(required))
    for _ in range(sweeps):
        old = alpha.copy()
        for j in range(len(alpha)):
            if gram[j, j] > 1e-14:
                alpha[j] = max(0.0, alpha[j] + (required[j] - gram[j] @ alpha) / gram[j, j])
        if np.max(np.abs(alpha - old)) < 1e-10:
            break
    delta = alpha @ directions
    return delta, bool(np.min(A @ delta - required) >= -1e-6)


def propose(policy, target, probes, layer, penalty, margin=0.1, cap=2.0):
    candidate = Policy(policy.weights)
    h1p, h2p = activations(policy, probes)
    if layer == "W3":
        features = np.column_stack((h2p, np.ones(len(h2p))))
        inverse = np.linalg.inv(np.eye(33) + penalty * features.T @ features / len(features))
        inverse_apply = lambda row: (inverse @ row.reshape(33, 5)).ravel()
        iterations = 1
    else:
        d2 = 1 - h2p ** 2
        blocks = np.stack([np.eye(32) + penalty *
                           (h1p.T @ (h1p * d2[:, j:j+1] ** 2)) / len(h1p)
                           for j in range(32)])
        inverses = np.linalg.inv(blocks)
        inverse_apply = lambda row: np.column_stack([
            inverses[j] @ row.reshape(32, 32)[:, j] for j in range(32)]).ravel()
        iterations = 8
    converged = False
    for iteration in range(iterations):
        A, margins = target_system(candidate, target, layer)
        if margins.min() >= margin * 0.99:
            converged = True
            break
        delta, solved = dual_projection(A, margin - margins, inverse_apply)
        norm = float(np.linalg.norm(delta))
        if not np.isfinite(delta).all() or norm == 0:
            break
        delta *= min(1.0, cap / norm)
        if layer == "W3":
            update = delta.reshape(33, 5)
            candidate.weights["W3"] += update[:32].astype(np.float32)
            candidate.weights["b3"] += update[32].astype(np.float32)
        else:
            candidate.weights["W2"] += delta.reshape(32, 32).astype(np.float32)
        if any(not np.isfinite(v).all() or np.abs(v).max() > 100 for v in candidate.weights.values()):
            return None, dict(reason="invalid_parameters")
    _, final_margins = target_system(candidate, target, layer)
    converged = bool(final_margins.min() >= margin * 0.99)
    delta_norm = float(np.sqrt(sum(np.sum((candidate.weights[k].astype(float) - v) ** 2)
                                  for k, v in policy.weights.items())))
    return candidate, dict(layer=layer, penalty=penalty, delta_norm=delta_norm,
                           minimum_margin=float(final_margins.min()), converged=converged,
                           iterations=iteration + 1)


def preservation(before, after, original):
    losses = [i for i, (b, a, o) in enumerate(zip(before, after, original))
              if a["total"] < max(b["total"], o["total"])]
    lost_clears = [i for i, (b, a, o) in enumerate(zip(before, after, original))
                  if (b["reason"] == "cleared" or o["reason"] == "cleared")
                  and a["reason"] != "cleared"]
    return dict(passed=not losses and not lost_clears, losses=losses, lost_clears=lost_clears,
                before_mean=float(np.mean([e["total"] for e in before])),
                after_mean=float(np.mean([e["total"] for e in after])),
                changed_trajectories=sum(b["actions"] != a["actions"] for b, a in zip(before, after)))


def check_bank(policy, bank):
    failures = []
    for item in bank:
        conditional = play(policy, item["seed"], prefix=item["prefix"])
        full = play(policy, item["seed"])
        if (not rescued(conditional, item) or full["total"] < item["accepted_return"]
                or item.get("world_target") and not world_corrected(full, item)):
            failures.append(item["id"])
    return failures


def verify_candidate(incumbent, candidate, target_case, bank, cases, before, original):
    # Scoring reconstructs its own policy from the persisted candidate, not solver claims.
    if policy_hash(candidate) == policy_hash(incumbent):
        return dict(accepted=False, reason="no_change"), None
    control = play(incumbent, target_case["seed"], prefix=target_case["prefix"])
    if serial_episode(control) != target_case["episode"] or rescued(control, target_case):
        raise ValueError("No-edit control failed to reproduce the exact original failure")
    conditional = play(candidate, target_case["seed"], prefix=target_case["prefix"])
    full = play(candidate, target_case["seed"])
    target_ok = rescued(conditional, target_case) and full["total"] > target_case["baseline_return"]
    if target_case.get("world_target"):
        target_ok = target_ok and world_corrected(full, target_case)
    report = dict(accepted=False, target_rescued=rescued(conditional, target_case),
                  target_full_return=full["total"], conditional=conditional, full=full)
    if not target_ok:
        return dict(**report, reason="target_not_corrected"), None
    lost_repairs = check_bank(candidate, bank)
    if lost_repairs:
        return dict(**report, reason="lost_prior_repair", lost_repairs=lost_repairs), None
    after = census(candidate, cases)
    gate = preservation(before, after, original)
    report.update(gate=gate, accepted=gate["passed"],
                  reason="accepted" if gate["passed"] else "preservation_regression")
    return report, after


def source_manifest():
    paths = [Path(__file__), ROOT / "vendor/weight_space_breakout/game.py",
             ROOT / "vendor/weight_space_breakout/policy.py"]
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def recover(run_dir, plan):
    """Committed cycle receipts are authoritative; state.json is only a cache."""
    checkpoint = "initial.npz"
    bank, history = [], []
    last_hash = plan["initial_policy_hash"]
    for cycle in range(plan["cycles"]):
        path = run_dir / f"cycle-{cycle:02d}/commit.json"
        if not path.exists():
            break
        receipt = read_json(path)
        if receipt["incumbent_hash"] != last_hash or receipt["cycle"] != cycle:
            raise ValueError("Broken checkpoint lineage")
        checkpoint = receipt["selected_checkpoint"]
        policy = Policy.load(run_dir / checkpoint)
        if policy_hash(policy) != receipt["selected_hash"]:
            raise ValueError("Committed checkpoint hash mismatch")
        if receipt["accepted"]:
            bank.extend(receipt.get("bank_entries", [receipt["bank_entry"]]))
        history.append(receipt)
        last_hash = receipt["selected_hash"]
    state_record = dict(next_cycle=len(history), checkpoint=checkpoint,
                        policy_hash=last_hash, repair_bank=bank)
    write_json(run_dir / "state.json", state_record)
    return state_record, history


def run(checkpoint, run_dir, *, resume=False, cycles=10, seed=9042026, patience=3, on_event=None,
        estimator=None, method="surgery", estimator_config=None):
    started = time.monotonic()
    run_dir = Path(run_dir)
    if not resume:
        run_dir.mkdir(parents=True, exist_ok=False)
        initial = Policy.load(checkpoint)
        save_policy(run_dir / "initial.npz", initial)
        plan = dict(protocol="experiment09-numerical-pilot-v3", cycles=cycles,
                    seed=seed, max_proposals_per_cycle=20, no_acceptance_stop=patience,
                    diagnosis_lookback=12, diagnostic_frontier_width=64,
                    world_search_depth=6, world_state_budget=2048, world_rollout_budget=512,
                    max_targets=2, mode="greedy", horizon=200,
                    layers=["W3", "W2"], penalties=[0., 10., 100., 1000., 10000.],
                    margin=0.1, step_norm_cap=2., max_w2_steps=8,
                    max_preservation_probes=2048,
                    initial_file_sha256=file_hash(checkpoint),
                    initial_policy_hash=policy_hash(initial), sources=source_manifest(),
                    runtime=dict(python=platform.python_version(), numpy=np.__version__,
                                 platform=platform.platform()),
                    cases=witnesses(), preservation="no return loss or lost board clear on all 96 configurations",
                    scope="Single numerical-estimator arm; not an LLM run or a three-arm comparison",
                    census_role="reused development gate, not independent held-out data")
        plan.update(method=method, estimator_config=estimator_config or {})
        if method == "gradient":
            plan.update(protocol="failure-conditioned-gradient-v1", layers=["adam-low", "adam-high"],
                        penalties=[0.1, 1., 10., 100., 1000.],
                        scope="Actual AdamW training on counterfactual paths with KL preservation; not direct surgery")
            plan["sources"]["hosted_pipeline/gradient.py"] = file_hash(ROOT / "hosted_pipeline/gradient.py")
        write_json(run_dir / "plan.json", plan)
        for relative in plan["sources"]:
            destination = run_dir / "source" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(ROOT / relative, destination)
        shutil.copyfile(ROOT / "docs/EXPERIMENT_09_RUN.md", run_dir / "source/protocol.md")
    else:
        plan = read_json(run_dir / "plan.json")
        if any(file_hash(ROOT / p) != digest for p, digest in plan["sources"].items()):
            raise ValueError("Source changed since frozen plan; use a new run directory")
        if plan["runtime"]["numpy"] != np.__version__ or plan["runtime"]["python"] != platform.python_version():
            raise ValueError("Runtime changed")
    initial = Policy.load(run_dir / "initial.npz")
    if policy_hash(initial) != plan["initial_policy_hash"]:
        raise ValueError("Initial checkpoint mismatch")
    cases = plan["cases"]
    original = census(initial, cases)
    write_json(run_dir / "baseline-census.json", original)
    current, history = recover(run_dir, plan)
    failed_streak = 0
    for h in reversed(history):
        if h["accepted"]:
            break
        failed_streak += 1
    incumbent = Policy.load(run_dir / current["checkpoint"])
    bank = current["repair_bank"]
    if on_event:
        on_event("initialized", dict(plan=plan, state=current, original=original))
    stop_reason = "cycle_budget"
    for cycle in range(current["next_cycle"], plan["cycles"]):
        if failed_streak >= plan["no_acceptance_stop"]:
            stop_reason = f"{plan['no_acceptance_stop']}_cycles_without_acceptance"
            break
        cycle_dir = run_dir / f"cycle-{cycle:02d}"
        cycle_dir.mkdir(parents=True, exist_ok=True)
        before = census(incumbent, cases, trace=True)
        inc_hash = policy_hash(incumbent)
        # Every natural failure is saved with its full trajectory before target selection.
        queue = [case_from_episode(e, inc_hash) for e in before if e["reason"] == "miss"]
        for item in queue:
            write_json(run_dir / "failures" / (item["id"] + ".json"), item)
        write_json(cycle_dir / "failure-queue.json", [dict(id=q["id"], seed=q["seed"],
                   status="captured_and_inspected", miss_step=q["target_step"],
                   prior_contacts=len(q["prefix"]) > 0) for q in queue])
        if not queue:
            stop_reason = "no_miss_failures"
            break
        previous_targets = {h.get("target_id") for h in history}
        fresh_queue = [q for q in queue if q["id"] not in previous_targets]
        if not fresh_queue:
            stop_reason = "all_current_failure_cases_attempted"
            break
        rng = np.random.default_rng(np.random.SeedSequence([plan["seed"], cycle]))
        target_case = fresh_queue[int(rng.integers(len(fresh_queue)))]
        episode = next(e for e in before if e["seed"] == target_case["seed"])
        targets, diagnostics = diagnose_worlds(incumbent, episode, target_case,
                                               [e for e in before if e["reason"] == "miss"],
                                               depth_limit=plan["world_search_depth"],
                                               state_budget=plan["world_state_budget"],
                                               rollout_budget=plan["world_rollout_budget"])
        write_json(cycle_dir / "diagnosis.json", dict(target_id=target_case["id"],
                   mechanism="reachable alternative world states across the full original trajectory; exact physics",
                   outcomes=diagnostics,
                   selected=[{k: v for k, v in t.items() if k not in ("observation", "sequence_observations")} for t in targets]))
        all_obs = np.concatenate([e["observations"] for e in before])
        idx = np.linspace(0, len(all_obs)-1, min(len(all_obs), plan["max_preservation_probes"]), dtype=int)
        probes = all_obs[idx]
        attempts, accepted, selected_path = [], False, current["checkpoint"]
        # This is an adaptive development census. There is no hidden independent gate claim.
        configs = itertools.product(targets, plan["penalties"], plan["layers"])
        for attempt, (target, penalty, layer) in enumerate(configs):
            if attempt >= plan["max_proposals_per_cycle"]:
                break
            candidate, estimate = (estimator or propose)(incumbent, target, probes, layer, penalty,
                                          plan["margin"], plan["step_norm_cap"])
            relative = f"cycle-{cycle:02d}/candidate-{attempt:02d}.npz"
            diagnostic_case = copy.deepcopy(target_case)
            diagnostic_case["world_target"] = True
            diagnostic_case["prefix"] = episode["actions"][:target["sequence_start"]]
            diagnostic_case["desired_world"] = target["desired_world"]
            if candidate is None:
                report = dict(accepted=False, reason="invalid_parameters")
            else:
                save_policy(run_dir / relative, candidate)
                candidate = Policy.load(run_dir / relative)
                report, after = verify_candidate(incumbent, candidate, diagnostic_case, bank,
                                                   cases, before, original)
                report["candidate_hash"] = policy_hash(candidate)
                if after is not None:
                    write_json(cycle_dir / f"census-{attempt:02d}.json", after)
            record = dict(attempt=attempt, target_step=target["step"], target_action=target["action"],
                          estimate=estimate, verification=report)
            write_json(cycle_dir / f"attempt-{attempt:02d}.json", record)
            attempts.append(dict(attempt=attempt, layer=layer, penalty=penalty,
                                 reason=report["reason"], target_rescued=report.get("target_rescued"),
                                 losses=report.get("gate", {}).get("losses", [])))
            print(json.dumps(dict(cycle=cycle, **attempts[-1])), flush=True)
            if on_event:
                on_event("proposal", dict(cycle=cycle, record=record))
            if report["accepted"]:
                accepted = True
                selected_path = relative
                break
        selected = Policy.load(run_dir / selected_path)
        receipt = dict(cycle=cycle, incumbent_hash=inc_hash, target_id=target_case["id"],
                       selected_checkpoint=selected_path, selected_hash=policy_hash(selected),
                       accepted=accepted, attempts=attempts,
                       status="accepted" if accepted else "no_verified_candidate",
                       natural_failures=len(queue), diagnosis_trials=diagnostics["continuation_rollouts"])
        if accepted:
            entry = copy.deepcopy(diagnostic_case)
            entry["accepted_return"] = report["target_full_return"]
            receipt["bank_entry"] = entry
            # Keep incidental corrections too: one edit can repair multiple edge cases.
            entries = [entry]
            for other in queue:
                if other["id"] == target_case["id"]:
                    continue
                result = next(a for a in after if a["seed"] == other["seed"])
                if world_corrected(result, other):
                    additional = copy.deepcopy(other)
                    additional.update(world_target=True, prefix=[], accepted_return=result["total"],
                                      incidental_to_target=target_case["id"])
                    entries.append(additional)
            receipt["bank_entries"] = entries
        # Commit precedes cache update. Restart recovers this receipt even if state write fails.
        write_json(cycle_dir / "commit.json", receipt)
        current, history = recover(run_dir, plan)
        bank = current["repair_bank"]
        incumbent = selected
        failed_streak = 0 if accepted else failed_streak + 1
        print(json.dumps(dict(cycle=cycle, accepted=accepted, retained_repairs=len(bank))), flush=True)
        if on_event:
            on_event("cycle_committed", dict(receipt=receipt, state=current))
    final = census(incumbent, cases)
    gate = preservation(original, final, original)
    if not gate["passed"] or check_bank(incumbent, bank):
        raise ValueError("Final independent replay failed preservation")
    write_json(run_dir / "final-census.json", final)
    gains = [a["total"] - b["total"] for a, b in zip(final, original)]
    summary = dict(protocol=plan["protocol"], status="completed", stop_reason=stop_reason,
                   cycles=len(history), proposals=sum(len(h["attempts"]) for h in history),
                   accepted_repairs=len(bank), accepted_checkpoints=sum(h["accepted"] for h in history),
                   initial_policy_hash=policy_hash(initial),
                   final_policy_hash=policy_hash(incumbent), checkpoint=current["checkpoint"],
                   baseline_mean=gate["before_mean"], final_mean=gate["after_mean"],
                   improved=sum(g > 0 for g in gains), regressed=sum(g < 0 for g in gains),
                   unchanged=sum(g == 0 for g in gains),
                   baseline_misses=sum(e["reason"] == "miss" for e in original),
                   final_misses=sum(e["reason"] == "miss" for e in final),
                   retained_repairs_verified=len(bank), elapsed_this_invocation_seconds=time.monotonic()-started,
                   pilot_success=sum(h["accepted"] for h in history) >= 3 and gate["after_mean"] > gate["before_mean"],
                   scope="96 exact legal configurations, greedy float32, 200 steps; development census, not transfer evidence")
    write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--seed", type=int, default=9042026)
    parser.add_argument("--patience", type=int, default=3)
    args = parser.parse_args()
    if not 1 <= args.cycles <= 100 or not 1 <= args.patience <= 100:
        parser.error("cycles and patience must be 1..100")
    if not args.resume and args.checkpoint is None:
        parser.error("--checkpoint is required for a new run")
    run(args.checkpoint, args.output, resume=args.resume, cycles=args.cycles, seed=args.seed,
        patience=args.patience)


if __name__ == "__main__":
    main()
