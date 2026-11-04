"""Verify the published evidence; optionally replay the frozen policies on CPU."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'vendor'))

import numpy as np
from safetensors.numpy import load_file
from weight_space_breakout.policy import Policy, SHAPES, apply_edits
from weight_space_breakout.repair_loop import comparison, fingerprint, from_sparse, passes_gate
from weight_space_breakout.failures import replay_failure

DATA = ROOT / 'results/qualified-repair-01'

def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

def require(condition, message):
    if not condition:
        raise ValueError(message)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--replay', action='store_true', help='Also simulate all published paired games.')
    args = parser.parse_args()
    manifest = read(ROOT/'CHECKSUMS.json')
    for name, expected in manifest.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT), 'Invalid checksum path')
        require(hashlib.sha256(path.read_bytes()).hexdigest() == expected, 'File changed: '+name)
    print(f'Verified {len(manifest)} file checksums.')

    report = read(DATA/'verifier-report.json')
    summary = read(DATA/'summary.json')
    baseline = Policy(load_file(ROOT/'checkpoints/baseline.safetensors'))
    repaired = Policy(load_file(ROOT/'checkpoints/repaired.safetensors'))
    require(fingerprint(baseline) == report['baseline_hash'] == summary['baseline_hash'], 'Baseline hash mismatch')
    require(fingerprint(repaired) == report['selected_hash'] == summary['repaired_hash'], 'Repaired hash mismatch')
    require(fingerprint(from_sparse(json.dumps(read(DATA/'baseline-policy.json')))) == fingerprint(baseline), 'Sparse baseline mismatch')
    require(fingerprint(from_sparse(json.dumps(read(DATA/'repaired-policy.json')))) == fingerprint(repaired), 'Sparse repair mismatch')
    patch = read(DATA/'patch.json')
    require(patch == report['edits'], 'Patch differs from verifier')
    require(fingerprint(apply_edits(baseline, patch)) == fingerprint(repaired), 'Patch did not produce checkpoint')
    changed = sum(int(np.count_nonzero(baseline.weights[k] != repaired.weights[k])) for k in SHAPES)
    require(changed == report['changed_parameters'] == 3, 'Unexpected changed-parameter count')

    with (DATA/'paired_scores.csv').open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    splits = [('practice_qualification','practice'), ('gate','gate'), ('final_audit','audit')]
    require(len(rows) == sum(report[key]['episodes'] for key,_ in splits), 'Unexpected score row count')
    for key, split in splits:
        expected = report[key]
        selected_rows = [r for r in rows if r['split'] == split]
        require([int(r['seed']) for r in selected_rows] == list(range(expected['seed_start'],expected['seed_end']+1)), 'Wrong seeds: '+split)
        before = np.array([float(r['baseline_return']) for r in selected_rows])
        after = np.array([float(r['repaired_return']) for r in selected_rows])
        require(np.array_equal(before,expected['baseline_returns']) and np.array_equal(after,expected['candidate_returns']), 'Scores differ: '+split)
        delta = after-before
        require(np.array_equal(delta,[float(r['difference']) for r in selected_rows]), 'Differences differ: '+split)
        measured = {'episodes':len(delta), 'baseline_return':float(before.mean()),
                    'candidate_return':float(after.mean()), 'gain':float(delta.mean()),
                    'paired_standard_error':float(delta.std(ddof=1)/np.sqrt(len(delta))),
                    'wins':int(np.sum(delta>0)), 'losses':int(np.sum(delta<0)), 'ties':int(np.sum(delta==0))}
        for field in ('episodes','baseline_return','candidate_return','gain','paired_standard_error'):
            require(np.isclose(measured[field],expected[field],atol=1e-12,rtol=0), f'{split}: incorrect {field}')
        for field,value in measured.items():
            require(np.isclose(value,summary['splits'][split][field],atol=1e-12,rtol=0), f'{split}: summary differs for {field}')
        print(f"{split:8s} n={len(delta):4d}: {before.mean():.6f} -> {after.mean():.6f}; gain={delta.mean():+.6f}; wins/losses/ties={measured['wins']}/{measured['losses']}/{measured['ties']}")
        if args.replay:
            actual = comparison(baseline, repaired, tuple(range(expected['seed_start'],expected['seed_end']+1)))
            require(actual['baseline_returns'] == expected['baseline_returns'], split+': baseline replay differs')
            require(actual['candidate_returns'] == expected['candidate_returns'], split+': repaired replay differs')
            require(actual['candidate_clear_rate'] == expected['candidate_clear_rate'], split+': clear rate differs')
            print('  All per-seed replay scores match.')

    require(passes_gate(report['practice_qualification'], changed), 'Practice qualification failed')
    require(passes_gate(report['gate'], changed) == report['accepted'] is True, 'Acceptance mismatch')
    trace = read(DATA/'trace-raw.json')
    require(trace['ok'] and trace['is_completed'] and not trace['errors'] and not trace['is_timeout'], 'Incomplete trace')
    require(trace['stop_condition'] == 'edit_submitted', 'Wrong stopping condition')
    require(trace['info']['repair_pipeline'] == report, 'Raw trace and report differ')
    final = next(node['message'] for node in reversed(trace['nodes']) if node['message']['role']=='assistant')
    calls = final['tool_calls']
    require(len(calls)==1 and calls[0]['name']=='repair_propose_edits', 'Wrong final tool')
    receipt = report['submission']
    require(json.loads(calls[0]['arguments']) == json.loads(receipt['json']), 'Final arguments differ from receipt')
    require(hashlib.sha256(receipt['json'].encode()).hexdigest() == receipt['sha256'], 'Receipt hash differs')
    require(json.loads(receipt['json'])['edits'] == patch['edits'], 'Submitted edits differ')
    require(report['candidate_replay']['prevented_recorded_miss'] is False, 'Diagnostic result differs')
    if args.replay:
        for policy,key in ((baseline,'baseline_replay'),(repaired,'candidate_replay')):
            require(replay_failure(policy,report['failure_case']) == report[key], 'Controlled replay differs: '+key)
        print('Controlled failure replays match; the recorded miss remains unprevented.')
    print('Verified: accepted three-parameter repair; average gain, with individual regressions.')

if __name__ == '__main__':
    main()
