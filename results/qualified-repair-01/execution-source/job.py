"""One HF-hosted execution of the unchanged released qualified-repair task."""
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

from huggingface_hub import HfApi, hf_hub_download
import verifiers.v1 as vf
from verifiers.v1.configs.agent import TimeoutConfig
from verifiers.v1.configs.client import EvalClientConfig
from weight_space_breakout.taskset import BreakoutConfig, BreakoutTaskset
from weight_space_breakout.editor_tools import EditorTaskConfig

OUT = Path('/tmp/repair-evidence')
OUT.mkdir(exist_ok=True)
REPO = os.environ['EVIDENCE_REPO']
REVISION = os.environ['INPUT_REVISION']

def save(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2), encoding='utf-8')

def read_input(name):
    return json.loads(Path(hf_hub_download(REPO, 'hf-job/input/'+name, revision=REVISION)).read_text())

async def main():
    plan = read_input('plan.json')
    args = read_input('environment-args.json')
    args['task'] = EditorTaskConfig(prepared_tools=True, qualified_proposals=True)
    task = BreakoutTaskset(BreakoutConfig(**args)).load()[0]
    config = vf.AgentConfig(
        model=plan['editor'], harness=vf.HarnessConfig(id='null'),
        runtime=vf.SubprocessConfig(),
        client=EvalClientConfig(base_url=os.environ['EDITOR_BASE_URL'], api_key_var='EDITOR_API_KEY'),
        sampling=vf.SamplingConfig(temperature=plan['temperature'], max_tokens=plan['max_tokens'],
                                   tool_choice='required', parallel_tool_calls=False),
        timeout=TimeoutConfig(setup=120, rollout=3600, finalize=120, scoring=600))
    save('agent-config.json', config.model_dump(mode='json'))
    save('plan.json', plan)
    save('environment-args.json', read_input('environment-args.json'))
    meta = {'compute_platform':'huggingface-jobs', 'inference_provider':'Prime Inference',
            'source_commit':plan['source_commit'], 'input_revision':REVISION,
            'python':platform.python_version(), 'numpy':importlib.metadata.version('numpy'),
            'verifiers':importlib.metadata.version('verifiers'),
            'environment_package':importlib.metadata.version('weight-space-breakout'),
            'started_at':time.time()}
    save('execution.json', meta)
    last = None
    def progress(trace):
        nonlocal last
        point = (len(trace.calls), trace.num_turns, len(trace.nodes), trace.stop_condition)
        if point != last:
            last = point
            print(json.dumps({'event':'trace_progress','calls':point[0], 'turns':point[1],
                              'nodes':point[2], 'stop':point[3]}), flush=True)
            (OUT/'trace-partial.json').write_text(trace.model_dump_json(), encoding='utf-8')
    try:
        async with vf.Agent(config) as agent:
            trace = await agent.run(task, on_trace=progress)
        (OUT/'trace-raw.json').write_text(trace.model_dump_json(indent=2), encoding='utf-8')
        report = trace.info.get('repair_pipeline')
        if report:
            save('verifier-report.json', report)
        meta.update(completed_at=time.time(), trace_id=trace.id, ok=trace.ok,
                    stop_condition=trace.stop_condition, num_turns=trace.num_turns,
                    accepted=report.get('accepted') if report else None)
        print(json.dumps({'event':'finished', **meta}), flush=True)
    except BaseException as error:
        save('job-error.json', {'type':type(error).__name__, 'message':str(error),
                               'traceback':traceback.format_exc()})
        raise
    finally:
        save('execution.json', meta)
        commit = HfApi().upload_folder(repo_id=REPO, revision=os.environ['EVIDENCE_BRANCH'],
            folder_path=OUT, path_in_repo='hf-job/output',
            commit_message='Preserve HF-hosted Qwen repair trace and unchanged verifier results')
        print(json.dumps({'event':'evidence_saved', 'revision':commit.oid}), flush=True)

asyncio.run(main())
