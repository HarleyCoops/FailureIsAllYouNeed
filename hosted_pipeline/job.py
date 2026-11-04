"""One HF Job: two original-base experiments, durable evidence, W&B, gated publication."""
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
from safetensors.numpy import save_file
import wandb

from experiment09 import loop as e
from hosted_pipeline.gradient import train_candidate
from scripts.audit_experiment09 import audit

CONFIG_PATH = Path(__file__).with_name("config.json")


def publishable(summary, verification):
    return (verification["status"] == "passed" and summary["final_mean"] > summary["baseline_mean"]
            and summary["regressed"] == 0 and summary["accepted_checkpoints"] > 0)


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows).encode()
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(data, mtime=0))
    else:
        path.write_bytes(data)


class Pipeline:
    def __init__(self, config, workspace):
        self.config, self.workspace = config, Path(workspace)
        self.api = HfApi()
        if self.api.whoami()["name"] != config["model_repo"].split("/")[0]:
            raise ValueError("Unexpected publication identity")
        self.stage = self.workspace / "dataset"
        self.stage.mkdir(parents=True, exist_ok=True)
        self.run_id = config["run_id"]
        self.progress = dict(run_id=self.run_id, status="running", source_commit=os.environ["SOURCE_COMMIT"],
                             base_model=config["base_model"], base_revision=config["base_revision"],
                             methods={}, releases={}, updated_at=time.time(),
                             training_space=config["training_space"], inference_space=config["inference_space"],
                             scope=config["evaluation_scope"])
        self.copied = {}

    def snapshot(self, arm, run_dir, status="running"):
        # Persist intermediate evidence each committed cycle. Large JSON is losslessly compressed.
        destination = self.stage / "runs" / self.run_id / "raw" / arm
        for path in sorted(Path(run_dir).rglob("*")):
            if not path.is_file() or path.suffix == ".tmp":
                continue
            signature = (path.stat().st_size, path.stat().st_mtime_ns)
            if self.copied.get(str(path)) == signature:
                continue
            relative = path.relative_to(run_dir)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".json":
                target = target.with_suffix(".json.gz")
                target.write_bytes(gzip.compress(path.read_bytes(), mtime=0))
            else:
                shutil.copyfile(path, target)
            self.copied[str(path)] = signature
        self.progress["updated_at"] = time.time()
        self.progress["methods"].setdefault(arm, {})["status"] = status
        e.write_json(self.stage / "latest.json", self.progress)
        e.write_json(self.stage / "runs" / self.run_id / "progress.json", self.progress)
        result = self.api.upload_folder(repo_id=self.config["dataset_repo"], repo_type="dataset",
                     folder_path=self.stage, commit_message=f"{self.run_id}: {arm} {status} evidence")
        print(json.dumps(dict(event="dataset_checkpoint", arm=arm, revision=result.oid)), flush=True)
        return result.oid

    def tables(self, arm, cycle, episodes, checkpoint_hash):
        rows, transitions = [], []
        for ep in episodes:
            trajectory_id = f"{self.run_id}:{arm}:{cycle}:{ep['seed']}:{checkpoint_hash[:12]}"
            common = dict(run_id=self.run_id, method=arm, cycle=cycle, seed=ep["seed"],
                          policy_hash=checkpoint_hash, trajectory_id=trajectory_id, split_role="development")
            rows.append(dict(**common, total_return=ep["total"], steps=ep["steps"], outcome=ep["reason"]))
            for t, action in enumerate(ep["actions"]):
                state, nxt = ep["states"][t:t+2]
                transitions.append(dict(**common, step=t, action=action,
                      reward=nxt["total"]-state["total"], state_json=json.dumps(state),
                      next_state_json=json.dumps(nxt), terminated=nxt["terminated"], truncated=nxt["truncated"]))
        folder = self.stage / "runs" / self.run_id / "tables" / arm
        write_rows(folder / "episodes" / f"cycle-{cycle:03d}.jsonl", rows)
        write_rows(folder / "transitions" / f"cycle-{cycle:03d}.jsonl.gz", transitions)

    def execute_arm(self, arm, baseline):
        config = self.config
        run_dir = self.workspace / arm
        run_key = hashlib.sha256(f"{self.run_id}:{arm}".encode()).hexdigest()[:8]
        wb = wandb.init(entity=config["wandb_entity"], project=config["wandb_project"],
                        id=run_key, name=f"{self.run_id}-{arm}", group=self.run_id,
                        job_type="failure-conditioned-training" if arm == "gradient" else "parameter-surgery",
                        config={**config, "method":arm, "source_commit":os.environ["SOURCE_COMMIT"]},
                        tags=[arm, "original-ppo-base", "fixed-physics", "development-census"])
        wb.define_metric("cycle")
        wb.define_metric("accepted/*", step_metric="cycle")
        wb.define_metric("proposal/index")
        wb.define_metric("proposal/*", step_metric="proposal/index")
        wb.define_metric("optimizer/global_step")
        wb.define_metric("optimizer/*", step_metric="optimizer/global_step")
        self.progress["methods"][arm] = dict(status="starting", curve=[], proposals=0, wandb_url=wb.url)
        metrics = self.progress["methods"][arm]
        optimizer_step = 0

        def callback(event, payload):
            nonlocal optimizer_step
            if event == "proposal":
                record = payload["record"]
                metrics["proposals"] += 1
                verification = record["verification"]
                log = {"proposal/index":metrics["proposals"], "proposal/accepted":int(verification["accepted"]),
                       "proposal/target_corrected":int(verification.get("target_rescued", False)),
                       "proposal/delta_norm":record["estimate"].get("delta_norm", 0.)}
                if "gate" in verification:
                    log.update({"proposal/mean_return":verification["gate"]["after_mean"],
                                "proposal/regressed_starts":len(verification["gate"]["losses"])})
                wb.log(log)
                for point in record["estimate"].get("history", []):
                    wb.log({"optimizer/global_step":optimizer_step+point["optimizer_step"],
                            **{f"optimizer/{k}":v for k,v in point.items() if k != "optimizer_step"}})
                optimizer_step += record["estimate"].get("steps", 0)
                return
            if event == "initialized":
                episodes = payload["original"]
                checkpoint = run_dir / "initial.npz"
                cycle, accepted = 0, False
                bank = []
            else:
                receipt, state = payload["receipt"], payload["state"]
                checkpoint = run_dir / state["checkpoint"]
                cycle, accepted = receipt["cycle"]+1, receipt["accepted"]
                bank = state["repair_bank"]
                episodes = e.census(e.Policy.load(checkpoint), e.witnesses())
            policy = e.Policy.load(checkpoint)
            mean = float(np.mean([ep["total"] for ep in episodes]))
            outcomes = Counter(ep["reason"] for ep in episodes)
            point = dict(cycle=cycle, mean_return=mean, misses=outcomes["miss"], clears=outcomes["cleared"],
                         time_limits=outcomes["time_limit"], accepted=accepted,
                         retained_distinct_starts=len({item["seed"] for item in bank}),
                         policy_hash=e.policy_hash(policy))
            metrics["curve"].append(point)
            wb.log({"cycle":cycle, "accepted/mean_return":mean, "accepted/misses":outcomes["miss"],
                    "accepted/clears":outcomes["cleared"], "accepted/time_limits":outcomes["time_limit"],
                    "accepted/retained_starts":point["retained_distinct_starts"],
                    "accepted/promotion":int(accepted)})
            if accepted or event == "initialized":
                artifact = wandb.Artifact(f"grid-policy-mlp-{arm}", type="model",
                                          metadata={"policy_hash":e.policy_hash(policy), "cycle":cycle})
                artifact.add_file(str(checkpoint), name="policy_weights.npz")
                wb.log_artifact(artifact, aliases=["latest-accepted" if accepted else "base", f"cycle-{cycle}"])
            self.tables(arm, cycle, episodes, e.policy_hash(policy))
            self.snapshot(arm, run_dir)

        try:
            summary = e.run(baseline, run_dir, cycles=config["cycles"], patience=config["patience"],
                            seed=config["seed"], method=arm, on_event=callback,
                            estimator=train_candidate if arm == "gradient" else None,
                            estimator_config=config["optimizer"] if arm == "gradient" else {})
            verification = audit(run_dir)
            metrics.update(summary=summary, audit=verification, status="audited")
            wb.summary.update({"final/mean_return":summary["final_mean"],
                               "final/baseline_mean":summary["baseline_mean"],
                               "final/gain":summary["final_mean"]-summary["baseline_mean"],
                               "final/regressions":summary["regressed"],
                               "final/misses":summary["final_misses"],
                               "final/accepted_checkpoints":summary["accepted_checkpoints"],
                               "final/audit_passed":True, "dataset":config["dataset_repo"]})
            if publishable(summary, verification):
                release = self.publish_model(arm, run_dir, summary, verification, wb.url)
                self.progress["releases"][arm] = release
                wb.summary.update({"publication/model":release["repo"], "publication/revision":release["revision"]})
            else:
                metrics["publication"] = "withheld: no verified improvement"
            self.snapshot(arm, run_dir, "completed")
            wb.finish()
        except BaseException as error:
            # Exception class only in shared status; never publish credentials in exception text.
            e.write_json(run_dir / "job-failure.json", dict(type=type(error).__name__, at=time.time()))
            self.snapshot(arm, run_dir, "failed")
            wb.finish(exit_code=1)
            raise

    def publish_model(self, arm, run_dir, summary, verification, wandb_url):
        if not publishable(summary, verification):
            raise ValueError("Publication gate failed")
        repo = self.config["model_repo"] if arm == "gradient" else self.config["surgery_model_repo"]
        if self.api.repo_exists(repo):
            raise ValueError("Refusing to replace an existing model; select a new run/repository")
        folder = self.workspace / f"export-{arm}"
        folder.mkdir()
        policy = e.Policy.load(run_dir / summary["checkpoint"])
        e.save_policy(folder / "policy_weights.npz", policy)
        shutil.copyfile(run_dir / "initial.npz", folder / "initial_baseline.npz")
        tensors = {}
        for i in (1,2,3):
            tensors[f"fc{i}.weight"] = np.ascontiguousarray(policy.weights[f"W{i}"].T)
            tensors[f"fc{i}.bias"] = np.ascontiguousarray(policy.weights[f"b{i}"])
        save_file(tensors, str(folder / "model.safetensors"))
        for name in ("config.json", "model.py"):
            shutil.copyfile(hf_hub_download(self.config["base_model"], name, revision=self.config["base_revision"]), folder / name)
        shutil.copyfile(e.ROOT / "vendor/weight_space_breakout/policy.py", folder / "inference.py")
        shutil.copyfile(e.ROOT / "LICENSE", folder / "LICENSE")
        for name in ("summary.json", "audit.json", "plan.json"):
            shutil.copyfile(run_dir / name, folder / name)
        card = f"""---
license: mit
base_model: {self.config['base_model']}
tags:
- reinforcement-learning
- failure-driven-learning
- {arm}
datasets:
- {self.config['dataset_repo']}
---
# {repo.split('/')[-1]}

Derived from the original PPO checkpoint at revision `{self.config['base_revision']}`.
Method: **{'AdamW backpropagation on failure-conditioned corrective paths with KL preservation' if arm == 'gradient' else 'direct constrained parameter edits'}**.
The physics and greedy action-selection rule were unchanged. This updates the
17,253-parameter game policy, not an editor language model.

Mean return across the exact 96-start development census: **{summary['baseline_mean']:.6f} → {summary['final_mean']:.6f}**.
Improved / regressed / equal: **{summary['improved']} / {summary['regressed']} / {summary['unchanged']}**.
Misses within 200 steps: **{summary['baseline_misses']} → {summary['final_misses']}**.
Check `audit.json` and the training dashboard for board clears versus time limits.
This is a reused development population; no unseen generalization or longer-horizon claim.

[W&B training curve]({wandb_url}) · [Trajectories](https://huggingface.co/datasets/{self.config['dataset_repo']}) · [Inference](https://huggingface.co/spaces/{self.config['inference_space']})

## Load

```python
from huggingface_hub import hf_hub_download
from inference import Policy  # inference.py from this pinned model revision
policy = Policy.load(hf_hub_download("{repo}", "policy_weights.npz"))
action = policy.act(observation, None, "greedy")
```

`model.safetensors` also contains PyTorch `fc1/fc2/fc3` keys for the original
`model.py` architecture. `initial_baseline.npz` preserves the exact starting tensors.
Final policy fingerprint: `{summary['final_policy_hash']}`.
Source commit: `{os.environ['SOURCE_COMMIT']}`.
"""
        (folder / "README.md").write_text(card, encoding="utf-8")
        self.api.create_repo(repo, private=False)
        commit = self.api.upload_folder(repo_id=repo, folder_path=folder,
                        commit_message=f"Publish audited {arm} self-improvement from original PPO base")
        loaded = e.Policy.load(hf_hub_download(repo, "policy_weights.npz", revision=commit.oid))
        if e.policy_hash(loaded) != summary["final_policy_hash"]:
            raise ValueError("Published model hash mismatch")
        return dict(repo=repo, revision=commit.oid, policy_hash=summary["final_policy_hash"],
                    method=arm, wandb_url=wandb_url)


def main():
    config = e.read_json(CONFIG_PATH)
    workspace = Path(os.environ.get("RUN_WORKSPACE", "/tmp/self-improved-run"))
    workspace.mkdir(parents=True, exist_ok=False)
    pipeline = Pipeline(config, workspace)
    revision = os.environ["INPUT_REVISION"]
    baseline = Path(hf_hub_download(config["dataset_repo"], "input/original-policy.npz",
                                   repo_type="dataset", revision=revision))
    provenance = e.read_json(hf_hub_download(config["dataset_repo"], "input/base-conversion.json",
                                           repo_type="dataset", revision=revision))
    if e.file_hash(baseline) != provenance["npz_sha256"]:
        raise ValueError("Base input hash mismatch")
    original_binary = hf_hub_download(config["base_model"], "pytorch_model.bin", revision=config["base_revision"])
    if e.file_hash(original_binary) != config["base_binary_sha256"]:
        raise ValueError("Original HF checkpoint hash mismatch")
    e.write_json(pipeline.stage / "runs" / config["run_id"] / "base-provenance.json", provenance)
    for arm in config["methods"]:
        pipeline.execute_arm(arm, baseline)
    pipeline.progress["status"] = "completed"
    pipeline.snapshot(config["methods"][-1], workspace / config["methods"][-1], "completed")
    print(json.dumps(dict(event="pipeline_completed", releases=pipeline.progress["releases"])), flush=True)


if __name__ == "__main__":
    main()
