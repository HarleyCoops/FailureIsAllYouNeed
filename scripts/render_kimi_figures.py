"""Render the separate Kimi study from saved records; never launch a repair."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/kimi'
ASSETS=ROOT/'assets'
BG='#f7f9fc'; INK='#14263d'; MUTED='#50657d'; GREEN='#007f67'; RED='#c1493b'
COLORS=['#8b98a8','#a45ac4','#2864c7','#007f67','#d58a16','#c1493b']
LAMBDAS=[0,1,10,100,1000,10000]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'figure.facecolor':BG,
    'axes.facecolor':BG,'axes.labelcolor':MUTED,'text.color':INK,'xtick.color':MUTED,
    'ytick.color':MUTED,'axes.edgecolor':'#dbe3ed','axes.spines.top':False,
    'axes.spines.right':False,'svg.fonttype':'none','svg.hashsalt':'failure-is-all-you-need'})

def read(name):
    return json.loads((DATA/name).read_text())

def save(fig,name):
    fig.savefig(ASSETS/f'{name}.png',dpi=180,facecolor=BG)
    svg=ASSETS/f'{name}.svg'
    fig.savefig(svg,metadata={'Date':None})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text(encoding='utf-8').splitlines())+'\n',encoding='utf-8')
    plt.close(fig)

def label(fig,x,y,text,size=11,color=INK,weight='normal'):
    fig.text(x,y,text,fontsize=size,color=color,fontweight=weight,va='baseline')

def verify_source():
    provenance=read('provenance.json')
    for rec in provenance['result_files']:
        assert hashlib.sha256((DATA/rec['path']).read_bytes()).hexdigest()==rec['sha256'],rec['path']
    assert len([r for r in provenance['wandb_runs'] if r['group'].endswith('-sweep')])==18
    # MCP marks this retrieval sampled. Verify the returned points individually;
    # the complete local history, rather than an API coverage claim, drives plots.
    observed=read('wandb-monotonic-history.json')
    local={r['attempt']:r for r in read('results_monotonic_100.json')['history'] if 'attempt' in r}
    for row in observed['rows']:
        rec=local[row['attempt']]
        assert row['mean_return']==rec['return_after']
        assert bool(row['kept'])==rec['kept']
        assert np.isclose(row['probe_flip_frac'],rec['probe_flips']/rec['probe_size'])

def sweep_figure():
    fig=plt.figure(figsize=(12,5.8))
    label(fig,.06,.94,'KIMI BRAIN SURGERY  /  18 SAVED SWEEP RUNS',10,COLORS[2],'bold')
    label(fig,.06,.855,'Local fixes can compound into global damage.',22,INK,'bold')
    label(fig,.06,.79,'Three solver families × six λ values  ·  Every candidate was applied in these sweeps',11,MUTED)
    axes=[fig.add_axes([.065+i*.315,.25,.275,.40]) for i in range(3)]
    for ax,family,title in zip(axes,['nullspace','w2','fullrank'],['W3 · soft null-space','W2 · rank one','W2 · unrestricted']):
        for lam,color in zip(LAMBDAS,COLORS):
            h=read(f'results_{family}_{lam}.json')['history']
            vals=[h[0]['return_before']]+[r['return_after'] for r in h]
            ax.plot(range(len(vals)),vals,color=color,lw=1.7,marker='o',ms=3,label=f'λ = {lam:,}')
        ax.axhline(h[0]['return_before'],color=INK,ls=(0,(3,3)),alpha=.35,lw=.9)
        ax.set(title=title,xlabel='Applied repair',xticks=[0,2,4,6,8],ylim=(0,13),yticks=[0,4,8,12])
        ax.grid(axis='y',color='#dbe3ed',lw=.7); ax.set_axisbelow(True)
    axes[0].set_ylabel('Mean return · same 64 gate games')
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.115),ncol=6,frameon=False,fontsize=9)
    label(fig,.065,.065,'These are composition experiments. Falling scores were measured, then retained; rollback was tested separately.',9,MUTED)
    label(fig,.065,.025,'Source: pinned results + W&B summaries  ·  PPO-trained policy / greedy actions  ·  separate from the Qwen repair',8,MUTED)
    save(fig,'kimi-sweeps')

def monotonic_figure():
    h=[x for x in read('results_monotonic_100.json')['history'] if 'attempt' in x]
    base=h[0]['return_before']; incumbent=[base]; candidates=[]
    for r in h:
        assert np.isclose(r['return_before'],incumbent[-1])
        assert r['kept']==(r['return_after']>=r['return_before'])
        incumbent.append(r['return_after'] if r['kept'] else incumbent[-1])
        candidates.append(r['return_after'])
    assert np.all(np.diff(incumbent)>=0)
    fig=plt.figure(figsize=(12,5.7))
    label(fig,.06,.94,'KIMI BRAIN SURGERY  /  KEEP–ROLLBACK TIMELINE',10,COLORS[2],'bold')
    label(fig,.06,.855,'The retained policy improves on the measured gate.',21,INK,'bold')
    label(fig,.06,.79,'λ = 100  ·  3 of 4 candidates kept  ·  Gate mean 8.16 → 9.83',11,MUTED)
    ax=fig.add_axes([.075,.24,.54,.42])
    ax.plot(range(5),incumbent,color=GREEN,lw=2.5,marker='o',ms=7,label='Retained policy')
    for r in h:
        if not r['kept']:
            ax.plot([r['attempt'],r['attempt']],[r['return_after'],r['return_before']],color=RED,ls='--')
            ax.scatter([r['attempt']],[r['return_after']],marker='x',s=85,color=RED,label='Rejected candidate',zorder=5)
    ax.annotate('Rollback\n9.58 rejected',xy=(3,candidates[2]),xytext=(2.7,8.82),fontsize=10,color=RED,
        arrowprops={'arrowstyle':'-','color':RED})
    ax.set(xticks=range(5),xlabel='Repair attempt (0 = baseline)',ylabel='64-game gate mean',ylim=(7.9,10.25))
    ax.grid(axis='y',color='#dbe3ed',lw=.7); ax.set_axisbelow(True)
    ax.legend(frameon=False,loc='lower right',fontsize=9)
    label(fig,.68,.64,'ATTEMPT',9,MUTED,'bold'); label(fig,.785,.64,'PROBE FLIPS',9,MUTED,'bold'); label(fig,.92,.64,'FIXED',9,MUTED,'bold')
    for i,r in enumerate(h):
        y=.56-i*.082
        label(fig,.68,y,f'{r["attempt"]}  '+('keep' if r['kept'] else 'reject'),10,GREEN if r['kept'] else RED)
        label(fig,.80,y,f'{r["probe_flips"]/r["probe_size"]:.2%}',10)
        label(fig,.925,y,'yes' if r['fixed'] else 'no',10)
    label(fig,.075,.115,'Keeping an edit required nondecreasing gate mean. A local fix and low probe damage were not required.',9,MUTED)
    label(fig,.075,.063,'The solid line is reconstructed after rollback; W&B’s raw mean_return logs the candidate score.',9,MUTED)
    label(fig,.075,.022,'Source: results_monotonic_100.json  ·  Separate held-out evaluation is still necessary.',8,MUTED)
    save(fig,'kimi-monotonic')

def paired_records():
    out=[]
    for version,name,key,role in [
        ('v1','paired_heldout.json','per_seed_repaired','original 1,000-seed evaluation; includes earlier 64-seed pilot'),
        ('v3','paired_heldout_v3.json','per_seed_v3','reused old evaluation set'),
        ('v4 fresh','paired_v4_fresh.json','per_seed_v4','fresh set after old-set failures informed development')]:
        data=read('eval/'+name)
        seeds=sorted(map(int,data['per_seed_baseline']))
        assert set(map(int,data[key]))==set(seeds) and len(seeds)==1000
        baseline=np.array([data['per_seed_baseline'][str(s)] for s in seeds])
        candidate=np.array([data[key][str(s)] for s in seeds]); diff=candidate-baseline
        out.append({'version':version,'seed_start':seeds[0],'seed_end':seeds[-1],'games':len(seeds),
            'baseline_mean':float(baseline.mean()),'repaired_mean':float(candidate.mean()),
            'paired_gain':float(diff.mean()),'paired_se':float(diff.std(ddof=1)/np.sqrt(len(seeds))),
            'wins':int((diff>0).sum()),'losses':int((diff<0).sum()),'ties':int((diff==0).sum()),
            'evaluation_role':role,'source_file':'eval/'+name})
    wb=next(r for r in read('provenance.json')['wandb_runs'] if r['id']=='fg4fxovn')['summary']
    for rec,prefix in [(out[0],'v1'),(out[2],'v4_fresh')]:
        assert np.isclose(rec['paired_gain'],wb[prefix+'_diff'])
        assert '/'.join(str(rec[k]) for k in ['wins','losses','ties'])==wb[prefix+'_wlt']
    (DATA/'paired-summary.json').write_text(json.dumps(out,indent=2)+'\n',encoding='utf-8')
    with (DATA/'paired-summary.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(out[0]),lineterminator='\n'); writer.writeheader();writer.writerows(out)
    tex=['% Generated by scripts/render_kimi_figures.py; no new rollouts.',r'\begin{center}\small',
         r'\begin{tabular}{lrrrrrr}',r'\toprule',r'Policy & Baseline & Repaired & Gain & Wins & Losses & Ties \\',r'\midrule']
    for r in out:
        tex.append(f'{r["version"]} & {r["baseline_mean"]:.3f} & {r["repaired_mean"]:.3f} & +{r["paired_gain"]:.3f} & {r["wins"]} & {r["losses"]} & {r["ties"]} '+r'\\')
    tex += [r'\bottomrule',r'\end{tabular}\end{center}',r'\newcommand{\VoneCI}{$[0.851,\,1.313]$}']
    assert read('eval/paired_heldout.json')['bootstrap_ci95_diff']==[.851,1.313]
    (ROOT/'docs/kimi-results-table.tex').write_text('\n'.join(tex)+'\n',encoding='utf-8')
    return out

def paired_figure(rows):
    fig=plt.figure(figsize=(12,5.6))
    label(fig,.06,.94,'KIMI BRAIN SURGERY  /  PAIRED EVALUATIONS',10,COLORS[2],'bold')
    label(fig,.06,.855,'Higher average return can coexist with regressions.',21,INK,'bold')
    label(fig,.06,.79,'1,000 games per comparison  ·  Versions use two different seed sets; results are not pooled',11,MUTED)
    ax=fig.add_axes([.2,.27,.73,.34])
    for i,r in enumerate(rows):
        y=2-i; left=0
        for k,color in [('wins',GREEN),('losses',RED),('ties','#8b98a8')]:
            n=r[k]; ax.barh(y,n,left=left,color=color,height=.56)
            if n>=70: ax.text(left+n/2,y,str(n),ha='center',va='center',color='white',weight='bold',fontsize=10)
            left+=n
        if r['losses']: ax.annotate(str(r['losses'])+' losses',xy=(r['wins']+r['losses']/2,y-.22),xytext=(r['wins']+50,y-.52),fontsize=8,color=RED,arrowprops={'arrowstyle':'-','color':RED})
        ax.text(-25,y,r['version']+f'  (+{r["paired_gain"]:.3f})',ha='right',va='center',weight='bold')
    ax.set(xlim=(0,1000),ylim=(-.75,2.65),xticks=[0,250,500,750,1000],yticks=[],xlabel='Games: improved / regressed / tied')
    ax.spines['left'].set_visible(False)
    label(fig,.075,.14,'v1 and v3: seeds 20000–20999. v4 fresh: seeds 30000–30999.',10,MUTED)
    label(fig,.075,.086,'Zero observed v1 losses is a finite-sample result. It does not establish universal non-regression.',10,MUTED)
    label(fig,.075,.028,'Source: copied per-seed JSON records; independently recomputed summaries, linked to source checkpoint hashes.',8,MUTED)
    save(fig,'kimi-paired')

if __name__=='__main__':
    verify_source(); sweep_figure(); monotonic_figure(); rows=paired_records(); paired_figure(rows)
    print(json.dumps(rows,indent=2))
