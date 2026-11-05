"""Provision a distinct cross-game experiment and submit its CPU Job once."""
import argparse
import json
import os
from pathlib import Path
import requests
from huggingface_hub import HfApi,get_token
from crossgame.run import ROOT,write

CONFIG=json.loads((ROOT/'config.json').read_text())


def prepare(output,commit):
    if len(commit)!=40 or any(c not in '0123456789abcdef' for c in commit):raise ValueError('Full source commit required')
    output.mkdir(parents=True,exist_ok=True)
    if (output/'prepared.json').exists():raise ValueError('Already prepared; inspect receipt')
    api=HfApi()
    if api.whoami()['name']!='HarleyCooper':raise ValueError('Unexpected HF account')
    if not os.environ.get('WANDB_API_KEY'):raise ValueError('Missing W&B credential')
    for key,kind in [('dataset','dataset'),('space','space'),('model','model')]:
        if api.repo_exists(CONFIG[key],repo_type=kind):raise ValueError('Output name already exists: '+CONFIG[key])
    stage=output/'dataset';stage.mkdir()
    (stage/'README.md').write_text(f"---\nlicense: cc-by-4.0\ntask_categories:\n- reinforcement-learning\n---\n# Asterix recursive learning experiment\n\n{CONFIG['scope']}\n\nRun protocol: [frozen source](https://github.com/HarleyCoops/FailureIsAllYouNeed/blob/{commit}/docs/ASTERIX_REGRESSION_FEEDBACK.md).\n\nDevelopment and gate seeds are reused for selection. The final audit is evaluated after selection ends. All candidates and failed attempts are retained. Base weights and upstream material: Rikiya Takehi and collaborators, CC BY 4.0. Official MinAtar game: Kenny Young and Tian Tian, GPL-3.0.\n",encoding='utf-8')
    write(stage/'latest.json',dict(status='provisioning',run_id=CONFIG['run_id'],curve=[],releases={}))
    write(stage/'provenance.json',json.loads((ROOT/'assets/provenance.json').read_text()))
    api.create_repo(CONFIG['dataset'],repo_type='dataset',private=False)
    dataset_commit=api.upload_folder(repo_id=CONFIG['dataset'],repo_type='dataset',folder_path=stage,commit_message='Register frozen Asterix cross-game protocol')
    space=output/'space';space.mkdir()
    docker=f'''FROM python:3.12.8-slim
WORKDIR /app
ADD https://github.com/HarleyCoops/FailureIsAllYouNeed/archive/{commit}.tar.gz /tmp/source.tar.gz
RUN tar -xzf /tmp/source.tar.gz --strip-components=1 -C /app && rm /tmp/source.tar.gz
RUN pip install --no-cache-dir -r hosted_pipeline/requirements.txt && pip install --no-cache-dir --no-deps minatar==1.0.15 && pip install --no-cache-dir torch==2.5.1+cpu --index-url https://download.pytorch.org/whl/cpu
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser
ENV HOME=/home/appuser HF_HOME=/tmp/hf WANDB_DIR=/tmp/wandb OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1 SOURCE_COMMIT={commit}
EXPOSE 7860
CMD ["python", "-m", "crossgame.space"]
'''
    (space/'Dockerfile').write_text(docker)
    (space/'README.md').write_text('---\ntitle: Asterix Recursive Learning\nemoji: 🧠\ncolorFrom: blue\ncolorTo: green\nsdk: docker\napp_port: 7860\nlicense: mit\n---\n# Asterix recursive learning\n\nNative MinAtar game, failure diagnosis, training evidence, and paired inference.\n',encoding='utf-8')
    api.create_repo(CONFIG['space'],repo_type='space',space_sdk='docker',space_hardware='cpu-basic',private=False)
    space_commit=api.upload_folder(repo_id=CONFIG['space'],repo_type='space',folder_path=space,commit_message='Build pinned Asterix experiment container')
    receipt=dict(source_commit=commit,dataset_revision=dataset_commit.oid,space_revision=space_commit.oid,config=CONFIG)
    write(output/'prepared.json',receipt);print(json.dumps(receipt,indent=2))


def launch(output):
    if (output/'submission-started.json').exists() or (output/'job.json').exists():raise ValueError('Submission already recorded; inspect before retry')
    receipt=json.loads((output/'prepared.json').read_text());api=HfApi()
    host=CONFIG['space'].lower().replace('/','-')
    health=requests.get(f'https://{host}.hf.space/health',timeout=30);health.raise_for_status()
    if health.json()['source_commit']!=receipt['source_commit']:raise ValueError('Source mismatch')
    image='registry.hf.space/'+host+':latest'
    write(output/'submission-started.json',dict(image=image,source_commit=receipt['source_commit']))
    job=api.run_job(image=image,command=['python','-u','-m','crossgame.run','--output','/tmp/asterix-run','--hosted'],
                    flavor='cpu-basic',timeout='60m',name=CONFIG['run_id'],
                    env={'SOURCE_COMMIT':receipt['source_commit'],'OPENBLAS_NUM_THREADS':'1','OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1'},
                    secrets={'HF_TOKEN':get_token(),'WANDB_API_KEY':os.environ['WANDB_API_KEY']},
                    labels={'experiment':'crossgame-asterix','run':CONFIG['run_id']})
    result=dict(job_id=job.id,url=job.url,image=image,source_commit=receipt['source_commit'])
    write(output/'job.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['prepare','launch']);p.add_argument('--output',type=Path,required=True);p.add_argument('--source-commit')
    a=p.parse_args()
    if a.operation=='prepare':prepare(a.output,a.source_commit)
    else:launch(a.output)
