"""Learn preservation constraints from replayed rejected-candidate regressions.

Only designated development seeds can enter this memory. A known better
incumbent trajectory supplies labels; the failed action is never a positive label.
"""
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np
from crossgame.asterix import Policy, play, serial


class RegressionMemory:
    def __init__(self, allowed_seeds, horizon, directory, writer, max_states=4096, max_cases=8):
        self.allowed=set(allowed_seeds);self.horizon=horizon;self.directory=Path(directory)
        self.writer=writer;self.max_states=max_states;self.max_cases=max_cases;self.entries={};self.events=[];self.teacher_cache={}

    def teacher(self, policy, seed):
        key=(policy.fingerprint(),seed)
        if key not in self.teacher_cache:
            if len(self.teacher_cache)>=4:self.teacher_cache.pop(next(iter(self.teacher_cache)))
            ep=play(policy,seed,self.horizon,trace=True)
            # No persistent simulator clones: observations and raw traces suffice.
            self.teacher_cache[key]={**serial(ep),'observations':ep['observations']}
        return self.teacher_cache[key]

    def ingest(self, incumbent, candidate, before, after, source, fixed_labels=False):
        if [r['seed'] for r in before]!=[r['seed'] for r in after]:raise ValueError('Unpaired regressions')
        event=dict(source=source,candidate_hash=candidate.fingerprint(),teacher_hash=incumbent.fingerprint(),cases=[])
        pairs=list(zip(before,after))
        if any(b['seed'] not in self.allowed for b,a in pairs):raise ValueError('Seed outside feedback training population')
        regressions=[(b,a) for b,a in pairs if a['total']<b['total'] or b['reason']=='time_limit' and a['reason']=='collision']
        regressions.sort(key=lambda pair:(-(pair[0]['total']-pair[1]['total']),pair[0]['seed']))
        event['all_regressions']=[dict(seed=b['seed'],before_return=b['total'],after_return=a['total'],
                                     before_reason=b['reason'],after_reason=a['reason']) for b,a in regressions]
        for b,a in regressions[:self.max_cases]:
            seed=b['seed']
            if seed not in self.allowed:raise ValueError('Seed outside feedback training population')
            if a['total']>=b['total'] and not (b['reason']=='time_limit' and a['reason']=='collision'):continue
            teacher=self.teacher(incumbent,seed)
            if {k:v for k,v in teacher.items() if k not in ('observations','worlds')}!=b:
                raise ValueError('Teacher replay changed')
            bad=play(candidate,seed,self.horizon,trace=True)
            if {k:v for k,v in serial(bad).items() if k!='worlds'}!=a:raise ValueError('Rejected replay changed')
            divergence=next((i for i,(old,new) in enumerate(zip(b['actions'],a['actions'])) if old!=new),None)
            if divergence is None:raise ValueError('Regression without action divergence')
            # Before the first different requested action, the native RNG and world match.
            if teacher['worlds'][divergence]!=bad['worlds'][divergence]:raise ValueError('Unmatched causal anchor')
            disagreements=[i for i,(obs,action) in enumerate(zip(teacher['observations'],teacher['actions']))
                           if candidate.act(obs)!=action]
            selected=sorted(set([divergence]+[disagreements[i] for i in np.linspace(0,len(disagreements)-1,min(32,len(disagreements)),dtype=int)]))
            for step in selected:
                obs=teacher['observations'][step];action=teacher['actions'][step]
                key=hashlib.sha256(obs.tobytes()).hexdigest()
                if key not in self.entries and len(self.entries)>=self.max_states:continue
                item=self.entries.setdefault(key,dict(observation=obs.copy(),action=action,count=0,seed=seed,step=step))
                # Labels are refreshed against the current accepted policy at train time.
                item['count']+=1
                if fixed_labels:
                    item['fixed_action']=action
            name=f"{len(self.events):03d}-seed-{seed}"
            self.writer(self.directory/(name+'.json.gz'),dict(source=source,teacher=serial(teacher),rejected=serial(bad),
                        first_divergence=divergence,training_steps=selected))
            event['cases'].append(dict(seed=seed,before_return=b['total'],after_return=a['total'],
                                       first_divergence=divergence,training_steps=selected))
        self.events.append(event)
        self.writer(self.directory/'index.json',dict(states=len(self.entries),events=self.events))
        return event

    def arrays(self, incumbent):
        items=sorted(self.entries.values(),key=lambda x:(-x['count'],x['seed'],x['step']))
        x=np.asarray([x['observation'] for x in items],dtype=np.float32).reshape(-1,600)
        y=np.asarray([item.get('fixed_action',incumbent.act(obs)) for item,obs in zip(items,x)],dtype=np.int64)
        counts=np.asarray([x['count'] for x in items],dtype=np.float32)
        return x,y,counts


def compatible_targets(primary, auxiliary):
    """Preserve primary correction and disclose conflicting auxiliary labels."""
    selected=[primary];labels={o.tobytes():a for o,a in zip(primary['observations'],primary['actions'])};conflicts=0
    for target in auxiliary:
        if any(o.tobytes() in labels and labels[o.tobytes()]!=a for o,a in zip(target['observations'],target['actions'])):
            conflicts+=1;continue
        selected.append(target)
        labels.update({o.tobytes():a for o,a in zip(target['observations'],target['actions'])})
    return selected,conflicts


def import_previous(config, memory, base, baseline, output, writer):
    """Replay pinned prior candidates. Never download their final-audit files."""
    from huggingface_hub import hf_hub_download
    parent=config['feedback']['parent'];aux=[];receipts=[]
    def fetch(path):return Path(hf_hub_download(parent['dataset'],path,repo_type='dataset',revision=parent['revision']))
    for proposal in parent['proposals']:
        path=fetch(proposal['path']);candidate=Policy.load(path)
        if candidate.fingerprint()!=proposal['policy_hash']:raise ValueError('Prior candidate fingerprint mismatch')
        after=[play(candidate,r['seed'],config['horizon']) for r in baseline]
        event=memory.ingest(base,candidate,baseline,after,'previous:'+proposal['path'])
        receipts.append(event)
        # Reconstruct the published candidate's original legal correction, not its failed behavior.
        cycle=proposal['path'].split('/')[0]
        failure=json.loads(gzip.decompress(fetch(cycle+'/failure.json.gz').read_bytes()))
        trial=proposal['correction'];prefix=failure['actions'][:trial['anchor']]+trial['sequence']
        ep=play(base,failure['seed'],config['horizon'],prefix=prefix,trace=True)
        if ep['total']!=trial['total'] or ep['steps']!=trial['steps']:raise ValueError('Prior corrective path does not replay')
        start=trial['anchor'];end=start+len(trial['sequence'])
        aux.append(dict(observations=ep['observations'][start:end],actions=trial['sequence']))
        writer(Path(output)/'imported-corrections'/f'{len(aux):02d}.json.gz',dict(proposal=proposal,seed=failure['seed'],trajectory=serial(ep)))
    writer(Path(output)/'feedback-import.json',dict(parent=parent,states=len(memory.entries),receipts=receipts))
    return aux


def import_audit_feedback(config, memory, incumbent, output, writer):
    """Promote the former audit into training; freeze its known better labels."""
    from huggingface_hub import hf_hub_download
    parent=config['continuation']
    def fetch(path):return Path(hf_hub_download(parent['dataset'],path,repo_type='dataset',revision=parent['revision']))
    reference=Policy.load(fetch('initial.npz'))
    old=Policy.load(fetch(parent['checkpoint']))
    if old.fingerprint()!=incumbent.fingerprint():raise ValueError('Wrong continuation checkpoint')
    with np.load(fetch('regression-training-memory.npz'),allow_pickle=False) as saved:
        for obs,action,count in zip(saved['observations'],saved['actions'],saved['counts']):
            key=hashlib.sha256(obs.tobytes()).hexdigest()
            memory.entries[key]=dict(observation=obs.copy(),action=int(action),count=int(count),seed=-1,step=-1,**({'fixed_action':int(action)} if config.get('objective')=='mean_return' else {}))
    inherited=len(memory.entries)
    before=json.loads(gzip.decompress(fetch('audit-base.json.gz').read_bytes()))
    after=json.loads(gzip.decompress(fetch('audit-final.json.gz').read_bytes()))
    expected=set(parent['regression_seeds'])
    actual={b['seed'] for b,a in zip(before,after) if a['total']<b['total'] or b['reason']=='time_limit' and a['reason']=='collision'}
    if actual!=expected:raise ValueError('Audit regression manifest mismatch')
    old_cap=memory.max_cases;memory.max_cases=len(expected)
    event=memory.ingest(reference,incumbent,before,after,'previous-audit-promoted-to-training',fixed_labels=True)
    memory.max_cases=old_cap
    if {x['seed'] for x in event['cases']}!=expected:raise ValueError('Not all audit failures were imported')
    writer(Path(output)/'audit-feedback-import.json',dict(parent=parent,inherited_states=inherited,
            imported_cases=len(event['cases']),total_states=len(memory.entries),event=event))
    return reference,{x['seed']:x for x in before}


def recovery_path(reference, incumbent, failure, horizon):
    """A whole known-better suffix is a legal recovery path, including timeouts."""
    teacher=play(reference,failure['seed'],horizon,trace=True)
    if teacher['total']<=failure['total'] or failure['reason']=='time_limit' and teacher['reason']!='time_limit':return None
    anchor=next((i for i,(a,b) in enumerate(zip(teacher['actions'],failure['actions'])) if a!=b),None)
    if anchor is None:return None
    # No world editing: replay actual prefix and the legal teacher suffix from reset.
    prefix=failure['actions'][:anchor]+teacher['actions'][anchor:]
    replay=play(incumbent,failure['seed'],horizon,prefix=prefix)
    if replay['total']!=teacher['total'] or replay['final']!=teacher['final']:raise ValueError('Recovery suffix changed its world')
    return dict(anchor=anchor,actions=teacher['actions'][anchor:],observations=teacher['observations'][anchor:],
                total=teacher['total'],steps=teacher['steps'],known_better_suffix=True)


def corrected_outcome(result, failure, recovery=False):
    if not recovery:return result['total']>failure['total'] and result['steps']>failure['steps']
    return (result['total']>failure['total'] and
            (failure['reason']!='time_limit' or result['reason']=='time_limit'))
