"""Recompute a completed experiment-09 run without proposing any new edits."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiment09 import loop as e


def audit(directory):
    global e
    root = Path(directory)
    plan = e.read_json(root / "plan.json")
    # Recompute historical runs with their archived optimizer and vendored engine.
    archived = root / "source/experiment09/loop.py"
    if archived.exists() and e.file_hash(archived) == plan["sources"]["experiment09/loop.py"]:
        import importlib.util
        spec = importlib.util.spec_from_file_location("frozen_experiment09", archived)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        e = module
    summary = e.read_json(root / "summary.json")
    if summary["status"] != "completed":
        raise ValueError("Audit requires a completed run")
    for relative, digest in plan["sources"].items():
        if e.file_hash(e.ROOT / relative) != digest:
            raise ValueError("Runtime source differs from frozen run: " + relative)
        if e.file_hash(root / "source" / relative) != digest:
            raise ValueError("Archived source hash mismatch")
    original_policy = e.Policy.load(root / "initial.npz")
    original = e.census(original_policy, plan["cases"])
    if original != e.read_json(root / "baseline-census.json"):
        raise ValueError("Baseline replay mismatch")
    before, policy, bank, timeline = original, original_policy, [], []
    proposals, reasons = 0, Counter()
    for cycle in range(summary["cycles"]):
        receipt = e.read_json(root / f"cycle-{cycle:02d}/commit.json")
        if receipt["incumbent_hash"] != e.policy_hash(policy):
            raise ValueError("Broken lineage")
        selected = e.Policy.load(root / receipt["selected_checkpoint"])
        if receipt["selected_hash"] != e.policy_hash(selected):
            raise ValueError("Selected checkpoint hash mismatch")
        for item in receipt["attempts"]:
            proposals += 1
            reasons[item["reason"]] += 1
        after = e.census(selected, plan["cases"])
        gate = e.preservation(before, after, original)
        if not gate["passed"] or e.check_bank(selected, bank):
            raise ValueError("Regression in committed checkpoint")
        added = receipt.get("bank_entries", []) if receipt["accepted"] else []
        bank.extend(added)
        if e.check_bank(selected, bank):
            raise ValueError("Saved correction does not reproduce")
        if not receipt["accepted"] and e.policy_hash(selected) != e.policy_hash(policy):
            raise ValueError("Rejected cycle changed weights")
        timeline.append(dict(cycle=cycle, accepted=receipt["accepted"],
                             mean_return=gate["after_mean"], new_correction_records=len(added),
                             retained_correction_records=len(bank),
                             retained_distinct_starting_configurations=len({b["seed"] for b in bank})))
        policy, before = selected, after
    if before != e.read_json(root / "final-census.json"):
        raise ValueError("Final replay mismatch")
    if e.policy_hash(policy) != summary["final_policy_hash"]:
        raise ValueError("Summary checkpoint mismatch")
    gains = np.array([a["total"]-b["total"] for a, b in zip(before, original)])
    if (float(np.mean([x["total"] for x in before])) != summary["final_mean"]
            or int((gains < 0).sum()) != summary["regressed"]):
        raise ValueError("Summary metrics mismatch")
    parameters = {k: dict(changed=int(np.count_nonzero(policy.weights[k] != v)),
                          l2=float(np.linalg.norm(policy.weights[k].astype(float)-v)))
                  for k, v in original_policy.weights.items()}
    result = dict(status="passed", baseline_trajectories_reproduced=96,
                  final_trajectories_reproduced=96, committed_cycles_replayed=len(timeline),
                  acceptance_decisions=sum(t["accepted"] for t in timeline),
                  retained_correction_records=len(bank),
                  retained_distinct_starting_configurations=len({b["seed"] for b in bank}),
                  proposals=proposals, outcomes=dict(reasons), timeline=timeline,
                  parameter_changes=parameters,
                  baseline_mean=summary["baseline_mean"], final_mean=summary["final_mean"],
                  improved_configurations=int((gains > 0).sum()),
                  regressed_configurations=int((gains < 0).sum()),
                  unchanged_configurations=int((gains == 0).sum()),
                  scope="Exact recomputation under frozen runtime; not a held-out or transfer test")
    e.write_json(root / "audit.json", result)
    print(__import__("json").dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    audit(parser.parse_args().run_directory)
