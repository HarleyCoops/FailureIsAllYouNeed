"""Submit the existing experiment to HF CPU; credentials stay in secret fields."""
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from huggingface_hub import HfApi, get_token

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'artifacts/qualified-repair-run-01'
REPO = 'HarleyCooper/weight-space-breakout-qualified-repair-01'
BRANCH = 'hf-job-20261104-01'
COMMIT = '0f4697c6c8cbfa48e96220c4e59881f5edfbd0b5'
if (HERE/'job-info.json').exists():
    raise RuntimeError('This attempt was already submitted; inspect it before another launch.')
api = HfApi()
assert api.whoami()['name'] == 'HarleyCooper'
plan = json.loads((OLD/'plan.json').read_text())
plan.update(created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            compute_platform='huggingface-jobs', inference_provider='Prime Inference',
            source_commit=COMMIT, previous_service_attempt='pyqeo6j8e329chclnlihsgu5',
            hosting_note='Released Prime verifier 0.4.5 executed through its Subprocess runtime on HF CPU; this is not a Prime-hosted evaluation.')
(HERE/'plan.json').write_text(json.dumps(plan,indent=2))
shutil.copy2(OLD/'state.json', HERE/'state.json')
shutil.copy2(OLD/'current-policy.json', HERE/'current-policy.json')
inputs = HERE/'input'
inputs.mkdir(exist_ok=True)
for src,dst in [(HERE/'plan.json','plan.json'), (HERE/'job.py','job.py'),
                (OLD/'round-00/environment-args.json','environment-args.json'),
                (OLD/'round-00/baseline-policy.json','baseline-policy.json')]:
    shutil.copy2(src,inputs/dst)
api.create_branch(REPO,branch=BRANCH)
uploaded = api.upload_folder(repo_id=REPO,revision=BRANCH,folder_path=inputs,
                            path_in_repo='hf-job/input',commit_message='Freeze unchanged repair protocol before HF-hosted execution')
# The subprocess pipe carries the saved key directly into the SDK secret field;
# neither its stdout nor the job secret values are printed or written to disk.
saved = subprocess.run(['wsl','-d','Ubuntu','--','/home/chris/.local/share/uv/tools/prime/bin/python',
    '-c','import json; from prime_cli.core import Config; c=Config(); print(json.dumps(dict(key=c.api_key,url=c.inference_url,team=c.team_id)))'],
    capture_output=True,check=True,text=True)
prime = json.loads(saved.stdout)
assert prime['key'] and not prime['team']
bootstrap = """import subprocess,sys,os,runpy
subprocess.check_call([sys.executable,'-m','pip','install','--quiet',
 'https://github.com/HarleyCoops/weight-space-breakout-pipeline/archive/COMMIT.zip',
 'huggingface-hub==1.27.0'])
from huggingface_hub import hf_hub_download
runpy.run_path(hf_hub_download(os.environ['EVIDENCE_REPO'],'hf-job/input/job.py',revision=os.environ['INPUT_REVISION']),run_name='__main__')
""".replace('COMMIT',COMMIT)
job = api.run_job(image='python:3.12-slim',command=['python','-u','-c',bootstrap],
    flavor='cpu-basic',timeout='70m',name='qualified-breakout-repair-01',
    env={'EVIDENCE_REPO':REPO,'EVIDENCE_BRANCH':BRANCH,'INPUT_REVISION':uploaded.oid,
         'EDITOR_BASE_URL':prime['url'],'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1'},
    secrets={'HF_TOKEN':get_token(),'EDITOR_API_KEY':prime['key']})
record={'job_id':job.id,'url':job.url,'compute_platform':'huggingface-jobs',
        'evidence_repo':REPO,'evidence_branch':BRANCH,'input_revision':uploaded.oid,
        'source_commit':COMMIT,'job_script_sha256':hashlib.sha256((HERE/'job.py').read_bytes()).hexdigest()}
(HERE/'job-info.json').write_text(json.dumps(record,indent=2))
print(json.dumps(record,indent=2))
