"""Read-only HF Space: live training curves and pinned-checkpoint inference."""
from functools import lru_cache
import os
from pathlib import Path
import time

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
import uvicorn

from experiment09 import loop as e

CONFIG = e.read_json(Path(__file__).with_name("config.json"))
app = FastAPI(title="GridPolicyMLP self-improvement")
_progress = (0., {})


def progress():
    global _progress
    if time.time() - _progress[0] > 15:
        try:
            path = hf_hub_download(CONFIG["dataset_repo"], "latest.json", repo_type="dataset")
            _progress = (time.time(), e.read_json(path))
        except Exception:
            if not _progress[1]:
                return dict(status="waiting", methods={}, releases={})
    return _progress[1]


@lru_cache(maxsize=8)
def load_policy(repo, revision, filename, repo_type):
    return e.Policy.load(hf_hub_download(repo, filename, revision=revision, repo_type=repo_type))


@app.get("/health")
def health():
    return dict(ok=True, source_commit=os.environ.get("SOURCE_COMMIT"), mode=os.environ.get("SPACE_MODE", "training"))


@app.get("/api/status")
def status():
    return progress()


@app.get("/api/episode")
def episode(seed: int = 5, variant: str = "gradient"):
    if not 0 <= seed <= 2147483647 or variant not in ("base", "gradient", "surgery"):
        raise HTTPException(400, "Choose a nonnegative seed and base, gradient or surgery")
    if variant == "base":
        revision = os.environ.get("INPUT_REVISION", "main")
        policy = load_policy(CONFIG["dataset_repo"], revision, "input/original-policy.npz", "dataset")
        model_repo = CONFIG["base_model"]
    else:
        release = progress().get("releases", {}).get(variant)
        if not release:
            raise HTTPException(409, "No audited model has been published for this method yet")
        revision, model_repo = release["revision"], release["repo"]
        policy = load_policy(model_repo, revision, "policy_weights.npz", "model")
        if e.policy_hash(policy) != release["policy_hash"]:
            raise HTTPException(503, "Model fingerprint verification failed")
    result = e.play(policy, seed)
    return dict(**result, variant=variant, model_repo=model_repo, revision=revision,
                policy_hash=e.policy_hash(policy), action_mode="greedy", horizon=200)


PAGE = r'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>GridPolicyMLP · Self improvement</title><style>
:root{font-family:system-ui,sans-serif;background:#f4f5f1;color:#20332c}body{max-width:1150px;margin:auto;padding:35px 22px}h1{font-size:38px;letter-spacing:-1.5px;margin-bottom:10px}h2{font-size:20px}p{line-height:1.6;color:#52635b}.eyebrow{color:#397f68;font-size:12px;letter-spacing:2px;font-weight:700}.cards,.games{display:grid;grid-template-columns:1fr 1fr;gap:18px}.card{background:white;border:1px solid #d8e2da;border-radius:16px;padding:22px;margin:12px 0}svg{width:100%;height:200px}a{color:#176e55}button{background:#176e55;color:white;border:0;padding:11px 18px;border-radius:8px;cursor:pointer}input,select{padding:10px;border:1px solid #a9b9ae;border-radius:7px;background:white}canvas{max-width:100%;width:330px;background:#15241e;border-radius:10px}small{overflow-wrap:anywhere;color:#607068}.metric{font-size:30px;font-weight:700}.tag{background:#e6efe9;padding:5px 10px;border-radius:20px;font-size:12px}#message{color:#985627}pre{white-space:pre-wrap;font-size:12px}footer{margin-top:35px;font-size:13px;color:#607068}@media(max-width:650px){.cards,.games{grid-template-columns:1fr}h1{font-size:29px}}
</style></head><body>
<div class="eyebrow">FAILURE IS ALL YOU NEED · ORIGINAL PPO BASE</div>
<h1>Learning from the worlds that failed.</h1>
<p>Two methods. The same starting weights, rules and physics. Gradient training learns from corrective paths; parameter surgery estimates direct edits. Only verified improvements become the next checkpoint.</p>
<span id="state" class="tag">Connecting…</span>
<div class="cards"><section class="card" id="gradient"></section><section class="card" id="surgery"></section></div>
<p><b>Accepted-policy curves:</b> rejected candidates never replace these weights. Evaluation covers all 96 legal starting configurations, with greedy actions and a 200-step limit. This is development data, not a test of unseen environments.</p>
<section class="card"><h2>Watch the actual policies</h2><p>Both games start from the same seed. Frames come from server-side inference using the published checkpoint—not a prerecorded animation.</p>
<label>Seed <input id="seed" type="number" value="5" min="0" max="2147483647" style="width:100px"></label>
<select id="variant"><option value="gradient">Gradient-trained model</option><option value="surgery">Surgery comparison</option></select>
<button id="play">Run paired games</button><p id="message"></p>
<div class="games"><div><h2>Original PPO</h2><canvas id="baseCanvas" width="400" height="400"></canvas><p id="baseScore"></p><small id="baseHash"></small></div><div><h2>Self-improved</h2><canvas id="newCanvas" width="400" height="400"></canvas><p id="newScore"></p><small id="newHash"></small></div></div></section>
<footer><a id="data" target="_blank">Trajectory dataset</a> · <a id="source" target="_blank">Frozen source</a><p>Zero misses within the horizon does not mean a board was cleared. Time limits and clears are reported separately. No action noise is added.</p></footer>
<script>
const $=id=>document.getElementById(id);let generation=0;
function chart(points){if(!points.length)return '<p>Waiting for the first evaluation.</p>';const ys=points.map(x=>x.mean_return),lo=Math.min(...ys)-.5,hi=Math.max(...ys)+.5;const coords=points.map((p,i)=>`${40+380*i/Math.max(1,points.length-1)},${165-125*(p.mean_return-lo)/(hi-lo)}`).join(' ');return `<svg viewBox="0 0 450 200" aria-label="Accepted mean return curve"><path d="M40 20 V165 H430" fill="none" stroke="#b8c8bf"/><polyline points="${coords}" fill="none" stroke="#176e55" stroke-width="3"/><text x="4" y="25" fill="#607068" font-size="12">${hi.toFixed(1)}</text><text x="4" y="165" fill="#607068" font-size="12">${lo.toFixed(1)}</text><text x="160" y="192" fill="#607068" font-size="12">Repair cycle · ${points.at(-1).cycle}</text></svg>`;}
async function update(){try{const p=await(await fetch('/api/status')).json();$('state').textContent=p.status||'waiting';for(const method of ['gradient','surgery']){const m=p.methods?.[method]||{},c=m.curve||[],last=c.at(-1);$(method).innerHTML=`<h2>${method==='gradient'?'Failure-conditioned training':'Parameter surgery'}</h2><span class="tag">${m.status||'queued'}</span><div class="metric">${last?last.mean_return.toFixed(3):'—'} <small>mean return</small></div>${chart(c)}<p>${last?`${last.misses} misses · ${last.time_limits} time limits · ${last.clears} clears`:'Original checkpoint will be evaluated first.'}</p><p>${m.proposals||0} candidate proposals</p>${m.wandb_url?`<a href="${m.wandb_url}" target="_blank">W&B metrics, losses and checkpoints ↗</a>`:''}${p.releases?.[method]?`<p><a href="https://huggingface.co/${p.releases[method].repo}" target="_blank">Published model ↗</a></p>`:''}`;}$('data').href='https://huggingface.co/datasets/HarleyCooper/grid-policy-mlp-self-improved-trajectories';$('source').href='https://github.com/HarleyCoops/FailureIsAllYouNeed/tree/'+(p.source_commit||'codex/self-improved-pipeline');}catch(e){$('state').textContent='Waiting for dataset connection';}}
function draw(id,s){const c=$(id).getContext('2d'),u=40;c.fillStyle='#15241e';c.fillRect(0,0,400,400);c.strokeStyle='#263c31';for(let i=0;i<=10;i++){c.beginPath();c.moveTo(i*u,0);c.lineTo(i*u,400);c.stroke();c.beginPath();c.moveTo(0,i*u);c.lineTo(400,i*u);c.stroke();}c.fillStyle='#cfaa65';for(const [r,col]of s.bricks)c.fillRect(col*u+3,r*u+4,u-6,u-8);c.fillStyle='#78c7a0';c.fillRect((s.paddle-1)*u+2,8*u+12,3*u-4,14);c.fillStyle='#fff7e0';c.beginPath();c.arc(s.col*u+20,s.row*u+20,9,0,Math.PI*2);c.fill();}
$('play').onclick=async()=>{const id=++generation;try{$('message').textContent='Running paired inference…';const seed=Number($('seed').value),v=$('variant').value;const rs=await Promise.all([fetch(`/api/episode?seed=${seed}&variant=base`),fetch(`/api/episode?seed=${seed}&variant=${v}`)]);const es=await Promise.all(rs.map(r=>r.json()));if(rs.some(r=>!r.ok))throw Error(es.find(e=>e.detail)?.detail||'Inference not ready');$('message').textContent='Same initial state · unchanged physics';for(let i=0;i<Math.max(...es.map(e=>e.states.length));i++){if(id!==generation)return;for(let j=0;j<2;j++){const e=es[j],s=e.states[Math.min(i,e.states.length-1)],prefix=j?'new':'base';draw(prefix+'Canvas',s);$(prefix+'Score').textContent=`Step ${s.step} · Return ${s.total} · ${s.reason}`;$(prefix+'Hash').textContent=`Checkpoint ${e.policy_hash} · revision ${e.revision}`;}await new Promise(r=>setTimeout(r,65));}}catch(e){$('message').textContent=e.message;}};
update();setInterval(update,15000);
</script></body></html>'''


@app.get("/", response_class=HTMLResponse)
def home():
    return PAGE


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
