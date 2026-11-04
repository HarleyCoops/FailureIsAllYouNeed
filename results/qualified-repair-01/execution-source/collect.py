"""Collect a real HF execution and apply the existing trace validator unchanged."""
import json
from pathlib import Path
import shutil
import sys
from huggingface_hub import HfApi, hf_hub_download

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path.insert(0,str(ROOT))
from pipeline.control import validate_trace, save, read
from weight_space_breakout.repair_loop import from_sparse, fingerprint

job=read(HERE/'job-info.json')
api=HfApi()
status=api.inspect_job(job_id=job['job_id'])
if status.status.stage != 'COMPLETED':
    raise RuntimeError('Job status: '+status.status.stage)
revision=api.model_info(job['evidence_repo'],revision=job['evidence_branch']).sha
raw=HERE/'evidence'
raw.mkdir(exist_ok=True)
for item in api.list_repo_files(job['evidence_repo'],revision=revision):
    if item.startswith('hf-job/output/'):
        shutil.copy2(hf_hub_download(job['evidence_repo'],item,revision=revision),raw/Path(item).name)
trace=read(raw/'trace-raw.json')
execution=read(raw/'execution.json')
config=read(raw/'agent-config.json')
assert execution['source_commit']==job['source_commit']
assert execution['environment_package']=='0.4.5'
assert execution['numpy']=='2.4.1' and execution['verifiers']=='0.4.1.dev97'
assert config['runtime']['type']=='subprocess' and config['harness']['id']=='null'
assert all(config[key] is None for key in ('max_turns','max_input_tokens','max_output_tokens','max_total_tokens','max_turn_tokens'))
assert trace['id']==execution['trace_id']
plan=read(HERE/'plan.json')
assert read(raw/'plan.json')==plan
# This is explicit transport metadata for the HF job, never a fabricated Prime
# evaluation. Keep the unmodified framework trace alongside the normalized copy.
run_id='hf-job-'+job['job_id']
trace['run']={'id':run_id,'platform':'huggingface-jobs','url':job['url']}
messages=[n['message'] for n in trace['nodes']]
summary={'trace_id':trace['id'],'environment_id':plan['environment'],
         'compute_platform':'huggingface-jobs',
         'activity':{'model_calls':len(trace['calls']),
                     'model_turns':sum(m['role']=='assistant' for m in messages),
                     'tool_calls':sum(len(m.get('tool_calls') or []) for m in messages),
                     'tool_results':sum(m['role']=='tool' for m in messages)},
         'execution':{'has_error':bool(trace['errors']) or not trace['ok'],
                      'is_truncated':not trace['is_completed'] or trace['is_timeout']},
         'score':{'outcome':trace['stop_condition']},
         'tool_definitions':trace.get('tools',[])}
baseline=from_sparse(json.dumps(read(HERE/'input/baseline-policy.json')))
submission={'evaluation_id':run_id,'compute_platform':'huggingface-jobs','round_index':0,
            'baseline_hash':fingerprint(baseline),'job_id':job['job_id'],
            'job_url':job['url'],'evidence_revision':revision,'source_commit':job['source_commit']}
candidate,selected,report,editor=validate_trace(trace,summary,submission,plan,baseline,0)
folder=HERE/'round-00'
folder.mkdir(exist_ok=True)
for name,value in [('trace.json',trace),('trace-summary.json',summary),('submission.json',submission),
    ('baseline-policy.json',baseline.sparse()),('candidate-policy.json',candidate.sparse()),
    ('verifier-report.json',report),('editor-response.json',editor),
    ('failure-case.json',report['failure_case']),
    ('failure-replay.json',{'baseline':report['baseline_replay'],'candidate':report.get('candidate_replay')})]:
    save(folder/name,value)
feedback={'accepted':report['accepted'],'valid':report['valid'],'edits':report.get('edits'),
          'changed_parameters':report.get('changed_parameters',0),'outcome':editor['outcome']}
if report['valid']:
    feedback.update({k:report['gate'][k] for k in ('baseline_return','candidate_return','gain','paired_standard_error')})
item={'round':0,'accepted':report['accepted'],'baseline_hash':report['baseline_hash'],
      'candidate_hash':report.get('candidate_hash'),'selected_hash':report['selected_hash'],
      'evaluation_id':run_id,'trace_id':trace['id'],'feedback':feedback,'job_url':job['url']}
state=read(HERE/'state.json')
assert state['completed_rounds']==0
state.update(completed_rounds=1,current_hash=report['selected_hash'],history=[item],final_audit=report['final_audit'])
save(HERE/'state.json',state)
save(HERE/'current-policy.json',selected.sparse())
# Add a result locator; the original pre-execution plan is retained in evidence.
plan['job_url']=job['url']
save(HERE/'plan.json',plan)
public=HERE/'model-export/experiment/hf-job'
public.mkdir(parents=True,exist_ok=True)
for path in raw.iterdir():
    if path.name!='trace-partial.json':
        shutil.copy2(path,public/path.name)
for name in ('job.py','launch.py','collect.py','job-info.json'):
    shutil.copy2(HERE/name,public/name)
print(json.dumps({'accepted':report['accepted'],'changed_parameters':report.get('changed_parameters'),
                  'activity':summary['activity'],'gate':{k:v for k,v in report.get('gate',{}).items() if not isinstance(v,list)},
                  'audit':{k:v for k,v in report['final_audit'].items() if not isinstance(v,list)}},indent=2))
