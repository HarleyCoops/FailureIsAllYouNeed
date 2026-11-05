"""One bounded cross-game trial with separate development, gate and final audit."""
import argparse
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

from crossgame.asterix import Policy, play, serial, evaluate, compare, diagnose, train
from crossgame.feedback import RegressionMemory, compatible_targets, import_previous

ROOT = Path(__file__).resolve().parent


def write(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    data=json.dumps(obj,indent=2,allow_nan=False).encode()
    if path.suffix=='.gz':
        path.write_bytes(gzip.compress(data,mtime=0))
    else:
        tmp=path.with_suffix('.tmp');tmp.write_bytes(data);os.replace(tmp,path)


def protected(before,after):
    comparison=compare(before,after)
    return comparison['losses']==0 and comparison['lost_survivors']==0,comparison


def run(output, config, hosted=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    write(output/'config.json',config)
    base=Policy.load(ROOT/'assets/asterix-base.npz');current=base
    base.save(output/'initial.npz')
    groups={k:list(range(*config[k+'_seeds'])) for k in ['development','gate','audit']}
    assert not (set(groups['development'])&set(groups['gate']) or set(groups['audit'])&(set(groups['development'])|set(groups['gate'])))
    horizon=config['horizon']; bank=[]; history=[]; visited=set(); wb=api=None
    status=dict(run_id=config['run_id'],status='running',source_commit=os.environ.get('SOURCE_COMMIT'),
                scope=config['scope'],curve=[],releases={},config=config)
    if hosted:
        import wandb
        from huggingface_hub import HfApi
        api=HfApi()
        wb=wandb.init(entity=config['wandb_entity'],project=config['wandb_project'],
                      id=hashlib.sha256(config['run_id'].encode()).hexdigest()[:8],
                      name=config['run_id'],config=config,tags=['asterix','recursive-learning','native-physics'])
        wb.define_metric('cycle');wb.define_metric('accepted/*',step_metric='cycle')
        status['wandb_url']=wb.url

    def snapshot():
        write(output/'latest.json',status)
        if api:
            api.upload_folder(repo_id=config['dataset'],repo_type='dataset',folder_path=output,
                              commit_message=f"{config['run_id']}: {status['status']} evidence")

    original={k:evaluate(base,groups[k],horizon) for k in ['development','gate']}
    before=original
    memory=None;auxiliary=[]
    if config.get('feedback'):
        memory=RegressionMemory(groups['development']+groups['gate'],horizon,output/'regression-memory',write,
                                max_states=config['feedback']['max_states'])
        auxiliary=import_previous(config,memory,base,original['development']+original['gate'],output,write)
        status['feedback_states']=len(memory.entries)
    for name,rows in original.items():write(output/f'baseline-{name}.json.gz',rows)
    status['curve'].append(dict(cycle=0,development_mean=compare(before['development'],before['development'])['before_mean'],
                                gate_mean=compare(before['gate'],before['gate'])['before_mean'],accepted=False))
    snapshot()
    for cycle in range(config['cycles']):
        failed=[r for r in before['development'] if r['reason']=='collision' and (current.fingerprint(),r['seed']) not in visited]
        if not failed:break
        failed.sort(key=lambda r:(r['total'],r['steps'],r['seed']))
        failure=failed[0];visited.add((current.fingerprint(),failure['seed']))
        episode=play(current,failure['seed'],horizon,trace=True)
        # The unedited control must reproduce all saved actions and final RNG state.
        assert {k:v for k,v in serial(episode).items() if k!='worlds'}==failure
        directory=output/f'cycle-{cycle:02d}';directory.mkdir()
        write(directory/'failure.json.gz',serial(episode))
        targets,diagnosis=diagnose(current,episode,**config['search'])
        write(directory/'diagnosis.json.gz',diagnosis)
        probes=[]
        for seed in groups['development']:
            ep=play(current,seed,horizon,trace=True)
            if ep['reason']=='collision':
                write(directory/'failures'/f'{seed}.json.gz',serial(ep))
            probes.extend(ep['observations'][::max(1,len(ep['actions'])//64)])
        probes=np.asarray(probes,dtype=np.float32)
        accepted=False; attempts=[]
        for index,(target,lr,penalty,refinement) in enumerate(itertools.product(targets,config['learning_rates'],config['preservation_penalties'],range(config.get('feedback',{}).get('refinements',1)))):
            joint_targets,aux_conflicts=compatible_targets(target,auxiliary)
            feedback_states=len(memory.entries) if memory else 0
            candidate,losses=train(current,joint_targets,probes,lr=lr,penalty=penalty,
                                   steps=config['optimizer_steps'],seed=config['training_seed']+cycle,
                                   feedback=memory.arrays(current) if memory else None,
                                   feedback_weight=config['feedback']['weights'][refinement] if memory else 0.)
            candidate_path=directory/f'candidate-{index:02d}.npz';candidate.save(candidate_path)
            candidate=Policy.load(candidate_path)
            actual=play(candidate,failure['seed'],horizon)
            conditional=play(candidate,failure['seed'],horizon,prefix=failure['actions'][:target['anchor']])
            corrected=all(r['total']>failure['total'] and r['steps']>failure['steps'] for r in [actual,conditional])
            report=dict(index=index,lr=lr,penalty=penalty,loss_history=losses,corrected=corrected,
                        candidate_hash=candidate.fingerprint(),accepted=False,reason='target_not_corrected')
            after={name:evaluate(candidate,groups[name],horizon) for name in ['development','gate']}
            report.update(refinement=refinement,feedback_states_used=feedback_states,auxiliary_conflicts=aux_conflicts,joint_targets=len(joint_targets))
            if corrected:
                retained=all((lambda r:r['total']>=b['accepted_return'] and r['steps']>b['failure_steps'])(play(candidate,b['seed'],horizon)) for b in bank)
                if retained:
                    checks={name:protected(before[name],after[name]) for name in after}
                    report['comparison']={name:v[1] for name,v in checks.items()}
                    accepted=all(v[0] for v in checks.values()) and checks['development'][1]['gain']>0
                    report.update(accepted=accepted,reason='accepted' if accepted else 'regression_gate')
                else:report['reason']='lost_retained_repair'
            if memory and not accepted:
                feedback_event=memory.ingest(current,candidate,before['development']+before['gate'],
                                              after['development']+after['gate'],f'cycle-{cycle:02d}/candidate-{index:02d}')
                report['regression_feedback']=dict(regressions=len(feedback_event['all_regressions']),
                                                   traced_cases=len(feedback_event['cases']),states_after=len(memory.entries))
                status['feedback_states']=len(memory.entries)
            write(directory/f'attempt-{index:02d}.json',report);attempts.append(report)
            print(json.dumps(dict(cycle=cycle,attempt=index,reason=report['reason'])),flush=True)
            if wb:
                wb.log({'candidate/cycle':cycle,'candidate/accepted':int(accepted),
                        'candidate/final_ce':losses[-1]['ce'],'candidate/final_kl':losses[-1]['kl'],
                        'candidate/regression_margin_loss':losses[-1]['regression_margin_loss'],
                        'candidate/feedback_states':feedback_states})
            if accepted:
                bank.append(dict(seed=failure['seed'],failure_steps=failure['steps'],accepted_return=actual['total']))
                auxiliary.append(target)
                current=candidate;before=after;break
        current.save(output/'accepted.npz')
        if memory:
            mx,my,mc=memory.arrays(current)
            np.savez_compressed(output/'regression-training-memory.npz',observations=mx,actions=my,counts=mc)
        record=dict(cycle=cycle,accepted=accepted,policy_hash=current.fingerprint(),bank=bank.copy(),
                    selected_checkpoint=str(candidate_path.relative_to(output)) if accepted else None,
                    attempts=len(attempts),seed=failure['seed'])
        write(directory/'commit.json',record);history.append(record)
        point=dict(cycle=cycle+1,accepted=accepted,
                   development_mean=float(np.mean([r['total'] for r in before['development']])),
                   gate_mean=float(np.mean([r['total'] for r in before['gate']])))
        status['curve'].append(point)
        if wb:wb.log({'cycle':cycle+1,**{'accepted/'+k:v for k,v in point.items() if k!='cycle'}})
        snapshot()
    # One final audit after selection ends. Never feed these seeds back into training.
    audit_base=evaluate(base,groups['audit'],horizon);audit_final=evaluate(current,groups['audit'],horizon)
    write(output/'audit-base.json.gz',audit_base);write(output/'audit-final.json.gz',audit_final)
    audit_pass,audit_comparison=protected(audit_base,audit_final)
    final={name:evaluate(current,groups[name],horizon) for name in ['development','gate']}
    assert final==before,'Final independent replay mismatch'
    replay_before=original
    for receipt in history:
        if not receipt['accepted']:continue
        checkpoint=Policy.load(output/receipt['selected_checkpoint'])
        assert checkpoint.fingerprint()==receipt['policy_hash'],'Checkpoint identity mismatch'
        replay_after={name:evaluate(checkpoint,groups[name],horizon) for name in ['development','gate']}
        assert all(protected(replay_before[name],replay_after[name])[0] for name in replay_after),'Committed checkpoint regressed'
        for repair in receipt['bank']:
            replay=play(checkpoint,repair['seed'],horizon)
            assert replay['total']>=repair['accepted_return'] and replay['steps']>repair['failure_steps'],'Lost retained repair'
        replay_before=replay_after
    for name,rows in final.items():write(output/f'final-{name}.json.gz',rows)
    summary=dict(status='completed',accepted_checkpoints=sum(r['accepted'] for r in history),
                 proposals=sum(r['attempts'] for r in history),cycles=len(history),
                 base_hash=base.fingerprint(),final_hash=current.fingerprint(),
                 development=compare(original['development'],final['development']),
                 gate=compare(original['gate'],final['gate']),held_out_audit=audit_comparison,
                 final_replay_passed=True,retained_repairs=bank,scope=config['scope'],
                 regression_feedback_states=len(memory.entries) if memory else 0,
                 regression_feedback_events=len(memory.events) if memory else 0)
    publish=summary['accepted_checkpoints']>0 and audit_pass and audit_comparison['gain']>0
    summary['publishable']=publish;write(output/'summary.json',summary)
    current.save(output/'final.npz')
    if api and publish:
        export=output/'model';export.mkdir();current.save(export/'policy_weights.npz')
        base.save(export/'initial_baseline.npz');write(export/'summary.json',summary)
        from safetensors.numpy import save_file
        save_file({f'fc{i}.{kind}': np.ascontiguousarray(current.weights[f'W{i}'].T if kind=='weight' else current.weights[f'b{i}'])
                   for i in [1,2,3] for kind in ['weight','bias']},str(export/'model.safetensors'))
        card=f"---\nlicense: cc-by-4.0\ntags:\n- reinforcement-learning\n- minatar\n- asterix\n---\n# Asterix failure-conditioned continuation\n\nDerived from the Astra-authored Asterix network in RikiyaT/lm_understand-logs, credited to Rikiya Takehi and collaborators. Gradient-trained here under official MinAtar 1.0.15 rules.\n\n{config['scope']}\n\nDevelopment: {summary['development']}\n\nHeld-out audit: {audit_comparison}\n\n[Evidence](https://huggingface.co/datasets/{config['dataset']}) · [Simulator](https://huggingface.co/spaces/{config['space']})\n"
        (export/'README.md').write_text(card,encoding='utf-8')
        if api.repo_exists(config['model']):raise ValueError('Model output already exists')
        api.create_repo(config['model'],private=False)
        release=api.upload_folder(repo_id=config['model'],folder_path=export,commit_message='Publish audited Asterix continuation')
        status['releases']['improved']=dict(repo=config['model'],revision=release.oid,policy_hash=current.fingerprint())
    status.update(status='completed',summary=summary)
    snapshot()
    if wb:wb.summary.update(summary);wb.finish()
    print(json.dumps(summary,indent=2),flush=True)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--hosted',action='store_true')
    args=parser.parse_args()
    run(args.output,json.loads((ROOT/'config.json').read_text()),args.hosted)
