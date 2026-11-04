"""Verify live Space inference against every frozen final-audit trajectory."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download
import numpy as np
import requests
from safetensors.numpy import load_file

from experiment09 import loop as e

CONFIG = e.read_json(Path(__file__).with_name("config.json"))


def verify(method, output):
    api = HfApi()
    dataset = CONFIG["dataset_repo"]
    dataset_revision = api.dataset_info(dataset).sha
    manifest = e.read_json(hf_hub_download(dataset, "latest.json", repo_type="dataset", revision=dataset_revision))
    release = manifest["releases"][method]
    policy = e.Policy.load(hf_hub_download(release["repo"], "policy_weights.npz", revision=release["revision"]))
    if e.policy_hash(policy) != release["policy_hash"]:
        raise ValueError("Model fingerprint differs from release")
    tensors = load_file(hf_hub_download(release["repo"], "model.safetensors", revision=release["revision"]))
    for i in (1,2,3):
        np.testing.assert_array_equal(policy.weights[f"W{i}"], tensors[f"fc{i}.weight"].T)
        np.testing.assert_array_equal(policy.weights[f"b{i}"], tensors[f"fc{i}.bias"])
    path = f"runs/{manifest['run_id']}/raw/{method}/final-census.json.gz"
    archive = hf_hub_download(dataset, path, repo_type="dataset", revision=dataset_revision)
    episodes = json.loads(gzip.decompress(Path(archive).read_bytes()))
    url = "https://" + CONFIG["inference_space"].lower().replace("/", "-") + ".hf.space/api/episode"

    def check(expected):
        response = requests.get(url, params={"seed":expected["seed"], "variant":method}, timeout=90)
        response.raise_for_status()
        actual = response.json()
        matches = (actual["revision"] == release["revision"] and actual["policy_hash"] == release["policy_hash"]
                   and all(actual[key] == expected[key] for key in ("actions", "states", "total", "reason", "steps")))
        return dict(seed=expected["seed"], matched=matches, total=actual["total"],
                    steps=actual["steps"], reason=actual["reason"])
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(check, episodes))
    receipt = dict(status="passed" if all(r["matched"] for r in results) else "failed",
                   method=method, model_repo=release["repo"], model_revision=release["revision"],
                   policy_hash=release["policy_hash"], dataset_revision=dataset_revision,
                   live_space=CONFIG["inference_space"], cases=len(results),
                   matched=sum(r["matched"] for r in results),
                   safetensors_equal_numpy=True, comparisons=results)
    e.write_json(output, receipt)
    if receipt["status"] != "passed":
        raise ValueError("Live inference does not match saved trajectories")
    api.upload_file(repo_id=dataset, repo_type="dataset", path_or_fileobj=Path(output),
                    path_in_repo=f"runs/{manifest['run_id']}/deployment-{method}.json",
                    commit_message=f"Verify all 96 live {method} trajectories and published tensor formats")
    print(json.dumps({k:v for k,v in receipt.items() if k != "comparisons"}, indent=2))
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("method", choices=["gradient", "surgery"])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.method, args.output)
