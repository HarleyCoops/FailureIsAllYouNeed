"""Provision new HF surfaces and submit the pinned two-arm job; never print secrets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil

from huggingface_hub import HfApi, get_token, hf_hub_download
import numpy as np
import requests

from experiment09 import loop as e

CONFIG = e.read_json(Path(__file__).with_name("config.json"))


def save(root, key, value):
    e.write_json(root / key, value)


def prepare(root):
    import torch
    root.mkdir(parents=True, exist_ok=True)
    if (root / "inputs.json").exists():
        return e.read_json(root / "inputs.json")
    api = HfApi()
    if api.whoami()["name"] != "HarleyCooper":
        raise ValueError("Unexpected HF identity")
    if not os.environ.get("WANDB_API_KEY"):
        raise ValueError("W&B credential unavailable")
    for repo in (CONFIG["model_repo"], CONFIG["surgery_model_repo"]):
        if api.repo_exists(repo):
            raise ValueError("Output model name already exists: " + repo)
    dataset = CONFIG["dataset_repo"]
    if api.repo_exists(dataset, repo_type="dataset"):
        raise ValueError("Dataset exists without this launch receipt; inspect before proceeding")
    staged = root / "input"
    staged.mkdir(exist_ok=True)
    original = hf_hub_download(CONFIG["base_model"], "pytorch_model.bin", revision=CONFIG["base_revision"])
    if e.file_hash(original) != CONFIG["base_binary_sha256"]:
        raise ValueError("Original HF checkpoint hash mismatch")
    state = torch.load(original, map_location="cpu", weights_only=True)
    weights = {}
    for i in (1,2,3):
        weights[f"W{i}"] = state[f"fc{i}.weight"].numpy().T.copy()
        weights[f"b{i}"] = state[f"fc{i}.bias"].numpy().copy()
    policy = e.Policy(weights)
    e.save_policy(staged / "original-policy.npz", policy)
    if any(not np.array_equal(v, policy.weights[k]) for k, v in weights.items()):
        raise ValueError("Conversion changed original float32 values")
    save(staged, "base-conversion.json", dict(base_model=CONFIG["base_model"],
         base_revision=CONFIG["base_revision"], original_binary_sha256=e.file_hash(original),
         npz_sha256=e.file_hash(staged / "original-policy.npz"), policy_hash=e.policy_hash(policy),
         conversion="fcN.weight transposed to WN; fcN.bias to bN; exact float32 tensor equality verified",
         rounded_kimi_export_used=False))
    card = f"""---
license: mit
task_categories:
- reinforcement-learning
configs:
- config_name: episodes
  default: true
  data_files:
  - split: development
    path: runs/*/tables/*/episodes/*.jsonl
- config_name: transitions
  data_files:
  - split: development
    path: runs/*/tables/*/transitions/*.jsonl.gz
---
# GridPolicyMLP self-improvement trajectories

Original PPO → failure-conditioned AdamW training and direct-parameter-surgery comparison.
All episodes use unchanged physics, greedy actions, 96 legal starting configurations,
and a 200-step horizon. These are reused development examples, not held-out tests.

`input/` contains the exact base conversion. `runs/<id>/raw/<method>/` preserves
all failure records, counterfactual worlds, proposals, rejections, checkpoints,
frozen source, and acceptance receipts. Large JSON is losslessly gzip-compressed.
`tables/` contains queryable episode and transition rows. `latest.json` serves
the live training dashboard and immutable published model revisions.

[Training](https://huggingface.co/spaces/{CONFIG['training_space']}) ·
[Inference](https://huggingface.co/spaces/{CONFIG['inference_space']}) ·
[W&B](https://wandb.ai/{CONFIG['wandb_entity']}/{CONFIG['wandb_project']})
"""
    upload = root / "dataset-initial"
    upload.mkdir(exist_ok=True)
    shutil.copytree(staged, upload / "input", dirs_exist_ok=True)
    (upload / "README.md").write_text(card, encoding="utf-8")
    save(upload, "latest.json", dict(run_id=CONFIG["run_id"], status="provisioning", methods={}, releases={}))
    api.create_repo(dataset, repo_type="dataset", private=False)
    commit = api.upload_folder(repo_id=dataset, repo_type="dataset", folder_path=upload,
                              commit_message="Freeze exact original PPO input and two-arm experiment schema")
    receipt = dict(dataset=dataset, revision=commit.oid,
                   base_policy_hash=e.policy_hash(policy), config=CONFIG)
    save(root, "inputs.json", receipt)
    print(json.dumps(receipt, indent=2))
    return receipt


def spaces(root, source_commit):
    if len(source_commit) != 40 or any(c not in "0123456789abcdef" for c in source_commit):
        raise ValueError("A full immutable GitHub commit is required")
    inputs = e.read_json(root / "inputs.json")
    api = HfApi()
    receipts = e.read_json(root / "spaces.json") if (root / "spaces.json").exists() else {}
    for mode, repo in (("training", CONFIG["training_space"]), ("inference", CONFIG["inference_space"])):
        if repo in receipts:
            if receipts[repo]["source_commit"] != source_commit:
                raise ValueError("Frozen source changed; use a separate deployment")
            continue
        if api.repo_exists(repo, repo_type="space"):
            raise ValueError("Space exists without launch ownership receipt: " + repo)
        folder = root / (mode + "-space")
        folder.mkdir(exist_ok=True)
        docker = f"""FROM python:3.12.8-slim
WORKDIR /app
ADD https://github.com/{CONFIG['github_repo']}/archive/{source_commit}.tar.gz /tmp/source.tar.gz
RUN tar -xzf /tmp/source.tar.gz --strip-components=1 -C /app && rm /tmp/source.tar.gz
RUN pip install --no-cache-dir -r hosted_pipeline/requirements.txt && pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser HF_HOME=/tmp/hf WANDB_DIR=/tmp/wandb OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 SOURCE_COMMIT={source_commit}
EXPOSE 7860
CMD ["python", "-m", "hosted_pipeline.space"]
"""
        (folder / "Dockerfile").write_text(docker, encoding="utf-8")
        (folder / "README.md").write_text(f"""---
title: GridPolicyMLP {'Training' if mode == 'training' else 'Self Improved'}
emoji: {'📈' if mode == 'training' else '🧠'}
colorFrom: green
colorTo: yellow
sdk: docker
app_port: 7860
license: mit
---
# GridPolicyMLP — {mode}

Frozen source: [{source_commit}](https://github.com/{CONFIG['github_repo']}/tree/{source_commit}).
The training Space supplies the Job image and a read-only progress dashboard.
Optimization runs in HF Jobs. Inference uses immutable, audited model revisions.
[Trajectories](https://huggingface.co/datasets/{CONFIG['dataset_repo']}).
""", encoding="utf-8")
        api.create_repo(repo, repo_type="space", private=False, space_sdk="docker",
                        space_hardware="cpu-basic", space_variables=[
                            {"key":"SPACE_MODE", "value":mode},
                            {"key":"INPUT_REVISION", "value":inputs["revision"]}])
        commit = api.upload_folder(repo_id=repo, repo_type="space", folder_path=folder,
                                  commit_message="Deploy pinned shared pipeline container")
        receipts[repo] = dict(revision=commit.oid, source_commit=source_commit, mode=mode)
        save(root, "spaces.json", receipts)
        print(json.dumps(dict(space=repo, revision=commit.oid)), flush=True)


def launch(root):
    if (root / "job.json").exists() or (root / "submission-started.json").exists():
        raise ValueError("Submission already recorded or ambiguous; inspect existing job before retrying")
    api = HfApi()
    inputs, surfaces = e.read_json(root / "inputs.json"), e.read_json(root / "spaces.json")
    repo = CONFIG["training_space"]
    runtime = api.get_space_runtime(repo)
    if str(runtime.stage) not in ("RUNNING", "SpaceStage.RUNNING"):
        raise ValueError("Training container is not running yet: " + str(runtime.stage))
    source_commit = surfaces[repo]["source_commit"]
    host = repo.lower().replace("/", "-")
    response = requests.get(f"https://{host}.hf.space/health", timeout=30)
    response.raise_for_status()
    if response.json()["source_commit"] != source_commit:
        raise ValueError("Container source mismatch")
    image = "registry.hf.space/" + host + ":latest"
    save(root, "submission-started.json", dict(name=CONFIG["job_name"], image=image, source_commit=source_commit))
    job = api.run_job(image=image, command=["python", "-u", "-m", "hosted_pipeline.job"],
           flavor=CONFIG["job_flavor"], timeout=CONFIG["job_timeout"], name=CONFIG["job_name"],
           env={"INPUT_REVISION":inputs["revision"], "SOURCE_COMMIT":source_commit,
                "RUN_WORKSPACE":"/tmp/self-improved-run", "OPENBLAS_NUM_THREADS":"1",
                "OMP_NUM_THREADS":"1", "MKL_NUM_THREADS":"1"},
           secrets={"HF_TOKEN":get_token(), "WANDB_API_KEY":os.environ["WANDB_API_KEY"]},
           labels={"experiment":"self-improved", "run":CONFIG["run_id"]})
    receipt = dict(job_id=job.id, url=job.url, source_commit=source_commit, image=image,
                    input_revision=inputs["revision"], methods=CONFIG["methods"])
    save(root, "job.json", receipt)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=["prepare", "spaces", "launch"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-commit")
    args = parser.parse_args()
    if args.operation == "prepare":
        prepare(args.output)
    elif args.operation == "spaces":
        spaces(args.output, args.source_commit or "")
    else:
        launch(args.output)
