"""Read-only training dashboard and paired native Asterix simulator."""
import json
import os
from pathlib import Path
from functools import lru_cache
from fastapi import FastAPI,HTTPException
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
import uvicorn
from crossgame.asterix import Policy,play

CONFIG=json.loads(Path(__file__).with_name('config.json').read_text())
app=FastAPI()


@app.get('/health')
def health():return {'ok':True,'source_commit':os.environ.get('SOURCE_COMMIT'),'game':'asterix'}


@app.get('/api/status')
def status():
    try:return json.loads(Path(hf_hub_download(CONFIG['dataset'],'latest.json',repo_type='dataset')).read_text())
    except Exception:return {'status':'waiting','curve':[],'releases':{}}


@lru_cache(maxsize=4)
def model(repo,revision):return Policy.load(hf_hub_download(repo,'policy_weights.npz',revision=revision))


@app.get('/api/episode')
def episode(seed:int=5,variant:str='base'):
    if not 0<=seed<2**31 or variant not in ('base','improved'):raise HTTPException(400,'Invalid seed or policy')
    if variant=='base':
        policy=Policy.load(Path(__file__).with_name('assets')/CONFIG.get('initial_checkpoint','asterix-base.npz'));revision=CONFIG.get('continuation',{}).get('revision','upstream-pinned')
    else:
        release=status().get('releases',{}).get('improved')
        if not release:raise HTTPException(409,'No published candidate is available yet')
        policy=model(release['repo'],release['revision']);revision=release['revision']
        if policy.fingerprint()!=release['policy_hash']:raise HTTPException(503,'Checkpoint identity mismatch')
    ep=play(policy,seed,CONFIG['horizon'],trace=True)
    # Compact public frames omit the large internal RNG state; raw dataset keeps it.
    frames=[{k:s[k] for k in ['step','total','dead','player','entities']} for s in ep['worlds']+[ep['final']]]
    return dict(seed=seed,total=ep['total'],steps=ep['steps'],reason=ep['reason'],actions=ep['actions'],
                frames=frames,policy_hash=policy.fingerprint(),revision=revision)


PAGE='''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Asterix recursive learning</title>
<style>body{font:16px system-ui;background:#111925;color:#e5edf6;max-width:1100px;margin:auto;padding:32px}p{line-height:1.6;color:#abbcd0}a{color:#88d7ba}.grid{display:grid;grid-template-columns:1fr 1fr;gap:24px}section{background:#1b293b;border-radius:16px;padding:20px;margin:20px 0}canvas{width:100%;max-width:400px;background:#0a111b}button,input{padding:12px;border-radius:8px;border:0}button{background:#88d7ba;cursor:pointer}svg{width:100%;height:190px}pre{white-space:pre-wrap}small{word-break:break-all}@media(max-width:650px){.grid{grid-template-columns:1fr}}</style></head><body>
<p>FAILURE IS ALL YOU NEED · CROSS-GAME EXPERIMENT</p><h1>Can the loop learn a different game?</h1>
<p>Asterix: collect gold, avoid enemies. Original MinAtar rules and native sticky actions remain intact. We continue from the published run 03 checkpoint and optimize average return. Individual regressions remain visible and feed the learning memory.</p>
<section><h2 id="status">Connecting…</h2><div id="curve"></div><p id="numbers"></p><p id="links"></p><pre id="summary"></pre></section>
<section><h2>Watch the policies</h2><p>Green is the player, gold is treasure, red is danger. Each game uses the same starting seed. Published inference uses the declared average-return criterion; some games may score worse or end earlier.</p>
<input id="seed" value="5" type="number" min="0"><button onclick="playGames()">Run paired games</button><p id="message"></p>
<div class="grid"><div><h3>Starting checkpoint (run 03)</h3><canvas id="base" width="400" height="400"></canvas><p id="baseScore"></p></div><div><h3>Published candidate weights</h3><canvas id="improved" width="400" height="400"></canvas><p id="improvedScore"></p></div></div></section>
<p>Development and selection gate seeds are reused. The final audit is separate and runs after selection ends. This is a 1,000-step greedy-policy pilot, not a reproduction of the upstream 10,000-step sampled-action score. Base weights: Rikiya Takehi and collaborators, CC BY 4.0. Game: Kenny Young and Tian Tian, MinAtar 1.0.15, GPL-3.0.</p>
<script>
const $=x=>document.getElementById(x);let generation=0;
async function update(){const p=await(await fetch('/api/status')).json();$('status').textContent=p.status+(p.feedback_states!==undefined?' · '+p.feedback_states+' regression examples':'');const c=p.curve||[];if(c.length){const vals=c.flatMap(x=>[x.development_mean,x.gate_mean]),lo=Math.min(...vals)-1,hi=Math.max(...vals)+1;let lines='';for(const [key,color]of [['development_mean','#88d7ba'],['gate_mean','#e5bb72']]){const pts=c.map((x,i)=>`${30+430*i/Math.max(1,c.length-1)},${160-135*(x[key]-lo)/(hi-lo)}`).join(' ');lines+=`<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="3"/>`}$('curve').innerHTML=`<svg viewBox="0 0 500 190">${lines}<text x="20" y="185" fill="white">Accepted checkpoint cycle · green: development · gold: gate</text></svg>`;$('numbers').textContent=`Development ${c.at(-1).development_mean.toFixed(3)} · Gate ${c.at(-1).gate_mean.toFixed(3)} mean return`;}$('links').innerHTML=`<a href="https://huggingface.co/datasets/__DATASET__">Trajectories and evidence</a>`+(p.wandb_url?` · <a href="${p.wandb_url}">W&B curves</a>`:'');if(p.summary)$('summary').textContent=JSON.stringify(p.summary,null,2);}
function draw(id,s){const c=$(id).getContext('2d');c.fillStyle='#0a111b';c.fillRect(0,0,400,400);for(const e of s.entities){if(!e)continue;c.fillStyle=e[3]?'#e5bb72':'#e47c7c';c.fillRect(e[0]*40+5,e[1]*40+5,30,30);}c.fillStyle='#88d7ba';c.fillRect(s.player[0]*40+6,s.player[1]*40+6,28,28);$(id+'Score').textContent=`Step ${s.step} · Score ${s.total}${s.dead?' · Collision':''}`;}
async function playGames(){const n=++generation;$('message').textContent='Running simulator…';const seed=Number($('seed').value);const rs=await Promise.all(['base','improved'].map(v=>fetch(`/api/episode?seed=${seed}&variant=${v}`)));const es=await Promise.all(rs.map(r=>r.json()));$('message').textContent=rs[1].ok?'Paired live inference':es[1].detail;const valid=es.filter((e,i)=>rs[i].ok);for(let t=0;t<Math.max(...valid.map(e=>e.frames.length));t++){if(n!==generation)return;for(let i=0;i<valid.length;i++)draw(i?'improved':'base',valid[i].frames[Math.min(t,valid[i].frames.length-1)]);await new Promise(r=>setTimeout(r,35));}}
update();setInterval(update,20000);
</script></body></html>'''.replace('__DATASET__',CONFIG['dataset'])


@app.get('/',response_class=HTMLResponse)
def home():return PAGE


if __name__=='__main__':uvicorn.run(app,host='0.0.0.0',port=7860)
