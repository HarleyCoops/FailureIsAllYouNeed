import json,sys,time
from pathlib import Path
import numpy as np
from safetensors.numpy import load_file

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'vendor'))
from weight_space_breakout.policy import Policy
from weight_space_breakout.evaluation import evaluate,rollout
from weight_space_breakout.failures import replay_failure
from weight_space_breakout.repair_loop import fingerprint

started=time.monotonic()
report=json.loads((root/'results/qualified-repair-01/verifier-report.json').read_text())
policies={name:Policy(load_file(root/f'checkpoints/{name}.safetensors')) for name in ['baseline','repaired']}
out={'protocol':'Published float32 policy and simulator; greedy comparison on original audit seeds is exploratory, not a new held-out test.',
     'runtime':{'numpy':np.__version__,'python':sys.version,'platform':sys.platform,'dtype':'float32'},'audit_seeds':[4090000,4091023],'policies':{}}
for name,p in policies.items():
    rec={'hash':fingerprint(p),'nonzero_parameters':sum(np.count_nonzero(v) for v in p.weights.values()),'seed314159':{}}
    for mode in ['sample','greedy']:
        ep=rollout(p,314159,mode=mode,record=True)
        rec['seed314159'][mode]={k:v for k,v in ep.items() if k!='frames'}
        rec['seed314159'][mode]['actions']=[f['action'] for f in ep['frames'][1:]]
        print(name,mode,{k:v for k,v in ep.items() if k!='frames'},flush=True)
    e=evaluate(p,range(4090000,4091024),mode='greedy')
    rec['greedy_audit']={k:v for k,v in e.items() if k not in ['seeds','returns']}
    rec['greedy_per_seed'] = dict(zip(map(str,e['seeds']),e['returns']))
    rec['controlled_replay']=replay_failure(p,report['failure_case'])
    out['policies'][name]=rec
    print(name,'greedy audit',e['mean_return'],flush=True)

agree=total=0
for p in policies.values():
    for seed in range(4040000,4040032):
        for frame in rollout(p,seed,mode='sample',record=True)['frames'][:-1]:
            obs=frame['observation']
            agree+=int(policies['baseline'].probabilities(obs).argmax()==policies['repaired'].probabilities(obs).argmax())
            total+=1
out['practice_state_argmax_comparison']={'states':total,'same_argmax':agree,'different_argmax':total-agree,'scope':'Pooled sampled baseline/repaired practice trajectories; includes repeated observations.'}
out['elapsed_seconds']=time.monotonic()-started
path=root/'results/action-mode-check/summary.json'
path.write_text(json.dumps(out,indent=2,default=lambda v:int(v) if isinstance(v,np.integer) else str(v))+'\n')
print('argmax comparison',out['practice_state_argmax_comparison'],flush=True)
print('Saved',path.name,'elapsed',out['elapsed_seconds'],flush=True)
