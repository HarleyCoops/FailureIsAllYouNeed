"""Render README figures from immutable scores, tool receipts, and checkpoints."""
from __future__ import annotations
from collections import Counter
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
from safetensors.numpy import load_file

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/qualified-repair-01'
OUT=ROOT/'assets'
OUT.mkdir(exist_ok=True)
REPORT=json.loads((DATA/'verifier-report.json').read_text())
TRACE=json.loads((DATA/'trace-raw.json').read_text())
BG='#f7f9fc'; INK='#14263d'; MUTED='#51667b'; GRID='#dce4ed'
BLUE='#2864c7'; GREEN='#007f67'; RED='#c1493b'; GRAY='#8b98a8'; GOLD='#a26308'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'text.color':INK,
    'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,
    'axes.edgecolor':GRID,'axes.spines.top':False,'axes.spines.right':False,
    'figure.facecolor':BG,'axes.facecolor':BG,'savefig.facecolor':BG,
    'svg.fonttype':'none','svg.hashsalt':'FailureIsAllYouNeed-2026-11-03'})

def save(fig,name):
    fig.savefig(OUT/(name+'.png'),dpi=180,metadata={'Software':'FailureIsAllYouNeed / Matplotlib'})
    svg=OUT/(name+'.svg')
    fig.savefig(svg,metadata={'Date':None,'Creator':'FailureIsAllYouNeed / Matplotlib'})
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n',encoding='utf-8',newline='\n')
    plt.close(fig)

def label(fig,x,y,text,size=11,color=MUTED,weight='normal',**kw):
    return fig.text(x,y,text,fontsize=size,color=color,fontweight=weight,**kw)

def panel_title(ax,letter,title,subtitle):
    ax.text(0,1.15,letter,transform=ax.transAxes,color=BLUE,fontsize=11,fontweight='bold')
    ax.text(.06,1.15,title,transform=ax.transAxes,color=INK,fontsize=13,fontweight='bold')
    ax.text(0,1.06,subtitle,transform=ax.transAxes,color=MUTED,fontsize=10)

def audit_figure():
    data=REPORT['final_audit']
    before=np.array(data['baseline_returns']); after=np.array(data['candidate_returns']); delta=after-before
    wins=int(sum(delta>0)); losses=int(sum(delta<0)); ties=int(sum(delta==0)); n=len(delta)
    fig=plt.figure(figsize=(12,8))
    label(fig,.065,.953,'FAILURE IS ALL YOU NEED  /  INDEPENDENT AUDIT',10,BLUE,'bold')
    label(fig,.065,.897,'Three changed parameters. Measurable average gain.',21,INK,'bold')
    label(fig,.065,.858,'1,024 paired games  ·  Candidate frozen before evaluation  ·  Same simulator, fresh seeds',11)
    for x,title,value,detail in [(.065,'FAULTY BASELINE',f'{before.mean():.2f}','mean return'),
        (.33,'ACCEPTED REPAIR',f'{after.mean():.2f}','mean return'),
        (.61,'PAIRED IMPROVEMENT',f'+{delta.mean():.2f}',f"paired SE {data['paired_standard_error']:.3f}")]:
        label(fig,x,.797,title,9,MUTED,'bold'); label(fig,x,.742,value,27,GREEN if x>.5 else INK,'bold')
        label(fig,x+.13,.75,detail,10)
    ax=fig.add_axes([.075,.285,.40,.33])
    panel_title(ax,'A','Every paired outcome','Circle area represents the number of games')
    counts=Counter(zip(before.astype(int),after.astype(int)))
    for (a,b),count in sorted(counts.items(),key=lambda x:-x[1]):
        ax.scatter(a,b,s=count*7,color=GREEN if b>a else RED if b<a else GRAY,alpha=.7,edgecolor=BG,linewidth=.5)
    ax.plot([-.8,24],[ -.8,24],color=GRAY,linestyle=(0,(4,4)),linewidth=1,zorder=0)
    ax.set(xlim=(-.8,24),ylim=(-.8,24),xlabel='Baseline return',ylabel='Repaired return',xticks=[0,6,12,18,24],yticks=[0,6,12,18,24])
    ax.text(.04,.91,'Above line: better',transform=ax.transAxes,fontsize=10,color=GREEN)
    ax.text(.61,.06,'Below: worse',transform=ax.transAxes,fontsize=10,color=RED)
    ax.grid(color=GRID,alpha=.6,linewidth=.6); ax.set_axisbelow(True)
    handles=[ax.scatter([],[],s=count*7,color=GRAY,alpha=.7,label=str(count)) for count in (5,25,50)]
    ax.legend(handles=handles,title='Games / circle',loc='upper right',frameon=False,fontsize=8,title_fontsize=8,handletextpad=.8)
    ax=fig.add_axes([.585,.285,.35,.33])
    panel_title(ax,'B','Distribution of changes','Repaired return minus baseline return')
    values=np.arange(int(delta.min()),int(delta.max())+1)
    frequency=np.array([sum(delta==v) for v in values])
    ax.bar(values,frequency,width=.85,color=[GREEN if v>0 else RED if v<0 else GRAY for v in values],alpha=.9)
    ax.axvline(data['gain'],color=BLUE,linewidth=1.5,linestyle=(0,(4,3)))
    ax.text(.96,.92,f"Mean +{data['gain']:.2f}",transform=ax.transAxes,ha='right',color=BLUE,fontsize=10,fontweight='bold')
    ax.set(xlabel='Change in episode return',ylabel='Games',xlim=(values[0]-1,values[-1]+1))
    ax.grid(axis='y',color=GRID,alpha=.65,linewidth=.6); ax.set_axisbelow(True)
    bar=fig.add_axes([.075,.157,.86,.045]); bar.set_xlim(0,n); bar.set_ylim(0,1); bar.axis('off')
    start=0
    for count,color,name in [(wins,GREEN,'improved'),(losses,RED,'regressed'),(ties,GRAY,'tied')]:
        bar.barh(.5,count,left=start,height=.75,color=color)
        bar.text(start+count/2,.5,str(count),ha='center',va='center',color='white',fontsize=11,fontweight='bold'); start+=count
    label(fig,.075,.12,f'{wins/n:.1%} improved',11,GREEN,'bold')
    label(fig,.49,.12,f'{losses/n:.1%} regressed',11,RED,'bold')
    label(fig,.80,.12,f'{ties/n:.1%} tied',11,MUTED,'bold')
    label(fig,.075,.058,'Average improvement does not guarantee a better outcome on every seed. No audit games cleared the board.',9)
    label(fig,.075,.031,'Source: verifier-report.json  ·  audit seeds 4090000–4091023  ·  Qwen editor / HF-hosted verifier',8)
    save(fig,'final-audit')

def proposal_rows():
    messages=[n['message'] for n in TRACE['nodes']]
    results={m['tool_call_id']:m for m in messages if m['role']=='tool'}
    rows=[]
    for message in messages:
        for call in message.get('tool_calls') or []:
            if call['name']!='repair_propose_edits': continue
            row={'proposal_call':len(rows)+1,'tool_call_id':call['id'],'valid':False,'gain':None,'paired_se':None,'changed_parameters':None,'qualified':False,'source':'tool_result'}
            result=results.get(call['id'])
            if result is None:
                assert json.loads(call['arguments'])==json.loads(REPORT['submission']['json'])
                row.update(valid=True,gain=REPORT['practice_qualification']['gain'],
                    paired_se=REPORT['practice_qualification']['paired_standard_error'],
                    changed_parameters=REPORT['changed_parameters'],qualified=True,source='terminal_receipt_and_verifier')
            else:
                try: parsed=json.loads(result['content'])
                except ValueError: parsed={}
                if parsed.get('valid'):
                    row.update(valid=True,gain=parsed['practice_gain'],paired_se=parsed['practice_paired_standard_error'],
                        changed_parameters=parsed['changed_parameters'],qualified=parsed['practice_qualified'])
            if row['valid']:
                assert row['qualified']==bool(row['changed_parameters']>0 and row['gain']>.1 and row['gain']>2*row['paired_se'])
            rows.append(row)
    assert len(rows)==29 and sum(r['valid'] for r in rows)==28
    assert [r['proposal_call'] for r in rows if r['qualified']]==[29]
    with (DATA/'proposal_history.csv').open('w',newline='',encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]),lineterminator='\n'); writer.writeheader(); writer.writerows(rows)
    return rows

def search_figure():
    rows=proposal_rows(); fig=plt.figure(figsize=(12,6.1))
    label(fig,.065,.94,'FAILURE IS ALL YOU NEED  /  PRACTICE SEARCH',10,BLUE,'bold')
    label(fig,.065,.864,'The first qualifying proposal ended the search.',22,INK,'bold')
    label(fig,.065,.81,'29 proposal calls  ·  28 valid patches  ·  32 fixed practice games per valid patch',11)
    ax=fig.add_axes([.08,.26,.86,.43])
    ax.axhline(0,color=GRAY,linewidth=1)
    ax.axhline(.1,color=GOLD,linewidth=1,linestyle=(0,(3,4)))
    for row in rows:
        x=row['proposal_call']
        if not row['valid']:
            ax.axvspan(x-.32,x+.32,color=GRAY,alpha=.12)
            ax.plot(x,.02,'x',transform=ax.get_xaxis_transform(),color=RED,markersize=8)
            continue
        ax.errorbar(x,row['gain'],yerr=2*row['paired_se'],fmt='o',markersize=6 if not row['qualified'] else 9,
                    color=GREEN if row['qualified'] else BLUE,alpha=1 if row['qualified'] else .7,
                    elinewidth=1.4,capsize=2)
    ax.set(xlim=(.2,30.8),ylim=(-3,4.7),xlabel='Proposal call, in execution order',ylabel='Practice return gain',xticks=[1,5,10,15,20,25,29])
    ax.grid(axis='y',color=GRID,linewidth=.7); ax.set_axisbelow(True)
    ax.annotate('Proposal 29\nGain +2.59\nAutomatically frozen',xy=(29,rows[-1]['gain']),xytext=(22.8,3.2),
        color=GREEN,fontsize=10,fontweight='bold',arrowprops={'arrowstyle':'-','color':GREEN,'linewidth':1})
    label(fig,.08,.17,'Dots: mean paired gain. Whiskers: ±2 paired SE. Gold line: the +0.10 gain threshold.',10)
    label(fig,.08,.124,'Qualification also requires the lower whisker above zero and a real weight change. Call 25 was invalid.',10)
    label(fig,.08,.078,'These are exploratory proposals against one unchanged baseline—not cumulative deployed improvements.',10,INK,'bold')
    label(fig,.08,.032,'Source: full tool trace; final point confirmed by the committed receipt and verifier. Separate gate and audit follow.',8)
    save(fig,'proposal-search')

def patch_figure():
    before=load_file(ROOT/'checkpoints/baseline.safetensors'); after=load_file(ROOT/'checkpoints/repaired.safetensors')
    changes={name:np.count_nonzero(before[name]!=after[name]) for name in before}
    assert sum(changes.values())==3 and changes['W3']==2 and changes['b3']==1
    fig=plt.figure(figsize=(12,6.1))
    label(fig,.065,.94,'FAILURE IS ALL YOU NEED  /  EXACT PARAMETER INTERVENTION',10,BLUE,'bold')
    label(fig,.065,.865,'Three assignments in the output layer.',22,INK,'bold')
    label(fig,.065,.81,'3 / 17,253 parameters changed (0.0174%)  ·  All earlier-layer weights and biases stayed fixed',11)
    ax=fig.add_axes([.065,.28,.43,.41]); ax.axis('off'); ax.set_xlim(-.02,1)
    for x,w,title,detail in [(0,.23,'500','observation'),(.33,.20,'32','tanh hidden'),(.64,.20,'32','tanh hidden')]:
        ax.add_patch(FancyBboxPatch((x,.53),w,.31,boxstyle='round,pad=0.012,rounding_size=.025',facecolor='#e7edf5',edgecolor=GRID))
        ax.text(x+w/2,.73,title,ha='center',va='center',fontsize=21,fontweight='bold',color=INK)
        ax.text(x+w/2,.6,detail,ha='center',va='center',fontsize=9,color=MUTED)
    for x1,x2 in [(.24,.31),(.54,.62)]:
        ax.annotate('',xy=(x2,.69),xytext=(x1,.69),arrowprops={'arrowstyle':'->','color':GRAY,'linewidth':1.5})
    ax.text(.43,.31,'Unchanged feature computation',ha='center',fontsize=12,fontweight='bold',color=MUTED)
    ax.text(.43,.16,'W1, b1, W2, b2',ha='center',fontsize=11,color=MUTED)
    ax.text(.43,.02,'500 → 32 → 32 → 5 architecture',ha='center',fontsize=10,color=MUTED)
    ax=fig.add_axes([.555,.27,.38,.44]); ax.set_xlim(-10,10); ax.set_ylim(-.6,2.6)
    changes=[('W3[0,3]  ·  left',8,-8),('W3[0,4]  ·  right',-8,8),('b3[0]  ·  no-op',-3,0)]
    ax.axvline(0,color=GRID,zorder=0)
    for y,(name,old,new) in zip([2,1,0],changes):
        ax.annotate('',xy=(new,y),xytext=(old,y),arrowprops={'arrowstyle':'->','color':BLUE,'linewidth':2.2,'shrinkA':6,'shrinkB':6})
        ax.scatter([old],[y],s=95,facecolor=BG,edgecolor=GRAY,linewidth=2,zorder=3)
        ax.scatter([new],[y],s=95,color=BLUE,edgecolor=BG,linewidth=1,zorder=3)
        ax.text(old,y+.22,f'{old:+g}',ha='center',fontsize=10,color=MUTED)
        ax.text(new,y+.22,f'{new:+g}',ha='center',fontsize=10,color=BLUE,fontweight='bold')
        ax.text(-10,y-.32,name,fontsize=10,color=INK)
    ax.set(yticks=[],xticks=[-8,-4,0,4,8],xlabel='Parameter value'); ax.spines['left'].set_visible(False)
    label(fig,.555,.756,'Open: baseline  /  Filled: accepted',9)
    label(fig,.065,.16,'The sign flips change left/right logits. Raising b3[0] increases the no-op logit.',11,INK,'bold')
    label(fig,.065,.108,'This identifies the executed edit. Component ablations are still needed to attribute the performance gain.',10)
    label(fig,.065,.04,'Source: published safetensors and exact patch  ·  Qwen’s language-model weights were not updated',8)
    save(fig,'weight-edit')

if __name__=='__main__':
    audit_figure(); search_figure(); patch_figure()
    print('Rendered final-audit, proposal-search, and weight-edit in PNG and SVG.')
