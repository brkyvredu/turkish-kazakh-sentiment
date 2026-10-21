#!/usr/bin/env python3
"""SAUCIS step 06: XLM-R effective-batch/precision/checkpoint engineering pilot.

Not paper training, not model selection, not final-test evaluation. Uses only
step-02 training/validation memberships and step-04 pinned XLM token bank.
Existing steps 03/04/05-recovery are imported ONLY after exact source checksums.
No downloads, pip installs, output deletion, precision fallback or GPU switching.

The parent never creates a CUDA context. Each task/precision and each checkpoint
replay runs in a separate process on the explicitly authorized physical GPU.
Full XLM-R state + AdamW + scheduler + Python/NumPy/CPU/CUDA RNG are checkpointed.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import nullcontext
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import traceback
import zipfile

VERSION = '1.0.0'
HELPERS = {
    'linear': ('05_revision_train_linear_recovery.py', '957ffda653a635eb63ea3b3f940f78e350fd4177f90e20f769a2ec10a9603338'),
    'features': ('04_revision_build_features.py', '1365dd5beca0a63a9e7b95da4e836e16279c8ad6d55aac3849c02e3ed488d430'),
    'smoke': ('03_revision_model_smoke.py', 'e045b983a7719717a3dbcaf25ba83b2201be4cd340aca411af80e6d2caeb45b0'),
}
EFFECTIVE_BATCH = 32
MAX_LENGTH = 256
PILOT_CONTEXTS = ('main:joint', 'binary:joint')
APPROVED_FAMILIES = ('main', 'binary', 'domain')
_STOP = False

class PilotError(RuntimeError): pass
class PilotStopped(PilotError): pass

def need(condition, message):
    if not condition: raise PilotError(message)

def now(): return datetime.now(timezone.utc).isoformat()
def canonical(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False)
def digest(x): return hashlib.sha256(x.encode('utf-8')).hexdigest()
def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(4*1024*1024), b''): h.update(chunk)
    return h.hexdigest()
def read_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def atomic(path, write, binary=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+f'.partial_{os.getpid()}_{time.time_ns()}')
    with temp.open('wb' if binary else 'w', **({} if binary else {'encoding':'utf-8'})) as f:
        write(f); f.flush(); os.fsync(f.fileno())
    os.replace(temp, path)
def write_json(path, value): atomic(path, lambda f: f.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n'), False)
def safe(root, name):
    p = Path(name)
    need(not p.is_absolute() and '..' not in p.parts, 'Unsafe relative output path.')
    out = Path(root)/p
    need(out.resolve().is_relative_to(Path(root).resolve()), 'Path escapes output root.')
    return out

def load_helper(project, alias):
    name, expected = HELPERS[alias]; path = Path(project)/name
    need(path.is_file(), f'Required existing helper is missing: {name}; upload the original delivered file.')
    need(sha(path)==expected, f'Existing helper differs from its reviewed version: {name}; do not edit it to bypass the check.')
    modname = '_saucis_pilot_'+alias
    if modname in sys.modules: return sys.modules[modname]
    spec = importlib.util.spec_from_file_location(modname, path)
    need(spec is not None and spec.loader is not None, 'Cannot import checked helper.')
    module = importlib.util.module_from_spec(spec); sys.modules[modname]=module
    spec.loader.exec_module(module)
    return module

def stop_handler(signum, frame):
    global _STOP
    _STOP = True

def complete(root, identity, names):
    files = {}
    for name in names:
        path = safe(root, name); need(path.is_file(), 'Cannot complete missing pilot artifact.')
        files[name] = {'bytes':path.stat().st_size, 'sha256':sha(path)}
    write_json(Path(root)/'complete.json', {'identity_sha256':digest(canonical(identity)), 'files':files})

def completed(root, identity):
    p = Path(root)/'complete.json'
    if not p.exists(): return False
    record = read_json(p)
    need(record['identity_sha256']==digest(canonical(identity)), 'Completed pilot identity differs.')
    need(bool(record['files']), 'Empty pilot completion record.')
    for name, item in record['files'].items():
        f = safe(root, name)
        need(f.is_file() and f.stat().st_size==item['bytes'] and sha(f)==item['sha256'], 'Completed pilot artifact is missing/corrupt; no automatic deletion.')
    return True

def audit_linear_archive(report, trials, selections, attempts):
    """Audit aggregate evidence; deliberately not a claim to have loaded all model files."""
    need(report['state']=='LINEAR_MODELS_READY' and report.get('progress_verified') is True, 'Linear training is not fully ready.')
    need(report['completed_C_fits']==6000 and report['selected_classifier_count']==1200 and len(report['completed'])==600, 'Unexpected approved linear coverage.')
    need(report.get('pending_or_blocked_candidates')==[], 'Linear candidates remain blocked/pending.')
    need(report['final_test_predictions_created'] is False and report['xlmr_training_performed'] is False, 'Unexpected prior test/XLM activity in linear report.')
    need(len(trials)==6000 and len(selections)==1200, 'CSV counts differ from linear report.')
    key = lambda r: (int(r['seed']),r['context_id'],r['representation'],r['classifier'])
    grouped=defaultdict(list)
    for r in trials:
        need(r['state']=='CANDIDATE_READY' and r['converged']=='True', 'Nonconverged linear candidate in selection CSV.')
        need(math.isfinite(float(r['validation_score'])), 'Nonfinite validation score.')
        grouped[key(r)].append(r)
    need(len(grouped)==1200 and len({key(r) for r in selections})==1200, 'Duplicate or missing classifier jobs.')
    for r in selections:
        rows=grouped[key(r)]
        need(len(rows)==5 and {float(t['C']) for t in rows}=={.01,.1,1.,10.,100.}, 'Incomplete/duplicate C grid.')
        chosen=min(rows,key=lambda t:(-float(t['validation_score']),float(t['C'])))
        need(float(r['selected_C'])==float(chosen['C']) and abs(float(r['selection_score'])-float(chosen['validation_score']))<1e-12, 'Saved selection violates validation/C tie rule.')
    bad=[a for a in attempts if a['state']=='NONCONVERGED']
    need(len(bad)==report['nonconverged_attempt_count'], 'Attempt summary mismatch.')
    for a in bad:
        target=key(a)+(float(a['C']),)
        successes=[b for b in attempts if key(b)+(float(b['C']),)==target and b['state']=='CONVERGED']
        need(len(successes)==1, 'A failed optimization candidate lacks one converged result.')
    return {'candidate_fits':len(trials),'selected_models':len(selections),'selection_checks':len(selections),
            'nonconverged_attempts':bad,'all_failed_candidates_recovered':True,
            'model_weight_files_opened':False, 'scope':'report/CSV consistency; not independent replay of all trained estimators'}

def audit_linear_directory(folder, base):
    p=Path(folder); r=read_json(p/'linear_training_report.json'); plan=read_json(p/'training_plan.json')
    need(plan['root_spec']==read_json(p/'root_spec.json'), 'Linear root/training plan mismatch.')
    root=plan['root_spec']
    for k in ('protocol_id','corpus_sha256','split_report_sha256','feature_report_sha256','smoke_report_sha256','packages'):
        need(root[k]==base[k], 'Linear inputs differ from current corpus/split/features/environment.')
    need(r['runner_sha256']==HELPERS['linear'][1] and r['protocol_id']==base['protocol_id'], 'Unexpected linear run.')
    def rows(name):
        with (p/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    trials=rows('all_validation_trials.csv');selections=rows('selected_validation_results.csv')
    expected={(int(seed),context,rep,clf) for seed,ctxs in plan['all_contexts'].items() for context in ctxs
              for rep in ('word','char','minilm','labse','bge_m3') for clf in ('lr','svm')}
    observed={(int(x['seed']),x['context_id'],x['representation'],x['classifier']) for x in selections}
    need(observed==expected and len(expected)==1200,'Selected models do not cover the approved training-context matrix.')
    audited=audit_linear_archive(r,trials,selections,rows('optimizer_attempts.csv'))
    audited['report_sha256']=sha(p/'linear_training_report.json')
    return audited

def set_metadata(L, split, seed, set_id):
    with L.ro_db(Path(split)/f'seed_{seed}'/'memberships.local.sqlite3') as db:
        row=db.execute('SELECT role,task,n_rows,metadata_json FROM sets WHERE set_id=?',(set_id,)).fetchone()
    need(row is not None and row[0] in ('train','validation'), 'Pilot/future training plan refuses final-test set.')
    return {'set_id':set_id,'role':row[0],'task':row[1],'rows':row[2],'metadata':json.loads(row[3])}

def future_jobs(L,split,contexts):
    jobs=[]
    for seed,ctxs in sorted(contexts.items()):
        for name,ctx in ctxs.items():
            if name.split(':')[0] not in APPROVED_FAMILIES:continue
            tr=set_metadata(L,split,seed,ctx['train_set']);va=set_metadata(L,split,seed,ctx['validation_set'])
            steps=3*math.ceil(tr['rows']/EFFECTIVE_BATCH)
            for lr in (1e-5,2e-5):jobs.append({'seed':seed,'context_id':name,'learning_rate':lr,
                'train_rows':tr['rows'],'validation_rows':va['rows'],'epochs':3,'optimizer_steps':steps,
                'warmup_steps_proposed':math.ceil(.1*steps),'selection_rule':ctx['selection_metric'],
                'validation_languages':ctx['validation_languages'], 'task':tr['task']})
    need(len(jobs)==90, 'Expected 90 main/binary/domain XLM jobs, without low-resource XLM sweep.')
    return jobs

def pick_indices(part, per_stratum, seed):
    import numpy as np
    rng=np.random.default_rng(seed);indices=[]
    for lang in sorted(set(part.languages)):
        for y in sorted(set(part.targets)):
            ids=np.flatnonzero((part.languages==lang)&(part.targets==y))
            need(len(ids)>=per_stratum,'Pilot needs more development rows in a stratum.')
            indices.extend(rng.choice(ids,size=per_stratum,replace=False).tolist())
    indices=np.asarray(indices,dtype=np.int64);rng.shuffle(indices)
    need(len(np.unique(indices))==len(indices),'Repeated pilot source indices.')
    return indices

def aligned_tokens(bank,part,indices):
    import numpy as np
    payload=bank.read_rows(part.row_ids[indices])
    need(np.array_equal(payload['sample_ids'],part.sample_ids[indices].astype('S64')), 'Token/sample order mismatch.')
    need(np.array_equal(payload['text_sha256'],part.text_hashes[indices].astype('S64')), 'Token/text fingerprint mismatch.')
    return payload

def padded(tokens,pad,max_length=MAX_LENGTH):
    import numpy as np
    need(len(tokens)>0,'No tokens to pad.')
    ids=np.full((len(tokens),max_length),pad,np.int64);mask=np.zeros_like(ids)
    for i,v in enumerate(tokens):
        need(0<len(v)<=max_length,'Token length exceeds approved maximum; no extra truncation.')
        ids[i,:len(v)]=v;mask[i,:len(v)]=1
    return ids,mask

def prepare_data(L,F,split,features,fr,root,context,pad):
    import numpy as np
    tr=L.load_development(split,42,context['train_set']);va=L.load_development(split,42,context['validation_set'])
    L.validate_pair(tr,va,context)
    task=tr.task;nclass=len(L.task_labels(task))
    ix=pick_indices(tr,16 if nclass==3 else 24,90210)[:81]
    jx=pick_indices(va,4,90211)
    bank=F.FeatureBank(L.safe_path(features,fr['banks']['xlmr_tokens']['bank_directory']))
    spec=bank.spec
    need(spec['kind']=='xlmr_token_ids' and spec['max_length']==256 and spec['labels_read'] is False and spec['fit_on_corpus'] is False,'Not a fixed label-free XLM token bank.')
    a=aligned_tokens(bank,tr,ix);b=aligned_tokens(bank,va,jx)
    X,M=padded(a['token_ids'],pad);V,VM=padded(b['token_ids'],pad)
    counts=Counter(map(int,tr.targets));weights=np.array([len(tr.targets)/(nclass*counts[k]) for k in range(nclass)],np.float32)
    need(len(ix)==81 and set(tr.group_ids[ix]).isdisjoint(set(va.group_ids[jx])),'Pilot train/validation groups overlap or incomplete batch stream.')
    arrays=dict(input_ids=X,attention_mask=M,labels=tr.targets[ix],validation_input_ids=V,validation_attention_mask=VM,
        validation_labels=va.targets[jx],row_ids=tr.row_ids[ix],sample_ids=tr.sample_ids[ix].astype('S64'),group_ids=tr.group_ids[ix].astype('S64'),
        validation_sample_ids=va.sample_ids[jx].astype('S64'),class_weights=weights)
    path=Path(root)/'pilot_data.PRIVATE.npz'
    signature=digest(canonical({'train':tr.identity(),'validation':va.identity(),'indices':ix.tolist(),'val_indices':jx.tolist(),
                               'token_spec_sha256':fr['banks']['xlmr_tokens']['spec_sha256'],'max_length':256,'pad':pad}))
    if (Path(root)/'data_ready.json').exists():
        old=read_json(Path(root)/'data_ready.json');need(old['signature']==signature and old['sha256']==sha(path),'Pilot data changed/corrupt.')
    else:
        atomic(path,lambda f:np.savez(f,**arrays))
        write_json(Path(root)/'data_ready.json',{'signature':signature,'sha256':sha(path)})
    return {'context_id':context['context_id'],'task':task,'num_labels':nclass,'train_manifest':tr.identity(),
        'validation_manifest':va.identity(),'pilot_train_rows':81,'pilot_validation_rows':len(jx),
        'effective_step_sizes':[32,32,17],'class_weights_from_full_authorized_training':weights.tolist(),
        'data_signature':signature,'data_sha256':sha(path),'fixed_padded_length':256,
        'pilot_raw_tokens_truncated':int((a['token_lengths']>256).sum()),'test_read':False}


def rng_state(torch):
    import numpy as np
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

def restore_rng(torch,state):
    import numpy as np
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch_cpu'])
    if state['torch_cuda']:
        need(torch.cuda.is_available() and len(state['torch_cuda'])==torch.cuda.device_count(),'Checkpoint GPU count differs.')
        torch.cuda.set_rng_state_all(state['torch_cuda'])

def cpu_tree(x):
    import torch
    if torch.is_tensor(x):return x.detach().cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x

def tree_sha(value):
    import torch
    import numpy as np
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):
            t=x.detach().cpu().contiguous();h.update(('tensor:'+str(t.dtype)+':'+str(tuple(t.shape))).encode())
            h.update(t.reshape(-1).view(torch.uint8).numpy().tobytes());return
        if isinstance(x,np.ndarray):
            a=np.ascontiguousarray(x);h.update(('array:'+str(a.dtype)+':'+str(a.shape)).encode());h.update(a.tobytes());return
        if isinstance(x,dict):
            h.update(b'{')
            for k in sorted(x,key=lambda v:(type(v).__name__,repr(v))):visit(k);visit(x[k])
            h.update(b'}');return
        if isinstance(x,(tuple,list)):
            h.update(type(x).__name__.encode()+b'[')
            for v in x:visit(v)
            h.update(b']');return
        h.update((type(x).__name__+':'+repr(x)+';').encode())
    visit(value);return h.hexdigest()

def save_state(folder,name,state,signature):
    import torch
    folder=Path(folder);file=folder/(name+'.PRIVATE.pt');pointer=folder/(name+'.ready.json')
    need(not pointer.exists(), 'Refusing to overwrite a committed checkpoint.')
    started=time.monotonic();atomic(file,lambda f:torch.save(cpu_tree(state),f))
    record={'signature':signature,'sha256':sha(file),'file':file.name,'bytes':file.stat().st_size,
            'step':state['step'],'cursor':state['cursor'],'save_seconds':time.monotonic()-started}
    write_json(pointer,record);return record

def load_state(folder,name,signature):
    import torch
    pointer=read_json(Path(folder)/(name+'.ready.json'));path=safe(folder,pointer['file'])
    need(pointer['signature']==signature and path.is_file() and path.stat().st_size==pointer['bytes'] and sha(path)==pointer['sha256'],'Checkpoint provenance/checksum mismatch.')
    # Only own checksum-verified local checkpoint; never load arbitrary .pt files.
    state=torch.load(path,map_location='cpu',weights_only=False)
    need(state['signature']==signature and state['step']==pointer['step'] and state['cursor']==pointer['cursor'],'Checkpoint progress mismatch.')
    return state,pointer

def state_pack(torch,model,opt,sched,signature,step,cursor,diagnostics):
    return {'signature':signature,'model':model.state_dict(),'optimizer':opt.state_dict(),'scheduler':sched.state_dict(),
            'rng':rng_state(torch),'step':step,'cursor':cursor,'diagnostics':diagnostics}

def training_step(model,opt,sched,ids,mask,y,weights,microbatch,precision,device):
    """One optimizer update, with correct weighted denominator across all microbatches.

    Includes incomplete effective batches: no dividing all microbatch means by a
    fixed accumulation count, which would mis-weight classes/last short batches.
    """
    import torch
    import numpy as np
    y=torch.as_tensor(y,dtype=torch.long,device=device)
    weights=torch.as_tensor(weights,dtype=torch.float32,device=device)
    denominator=weights[y].sum();need(bool(torch.isfinite(denominator)) and denominator.item()>0,'Invalid class-weight denominator.')
    model.train();opt.zero_grad(set_to_none=True);loss_value=0.;microsteps=0
    if device.startswith('cuda'):torch.cuda.synchronize()
    start=time.monotonic()
    for lo in range(0,len(y),microbatch):
        hi=min(lo+microbatch,len(y))
        x=torch.as_tensor(np.asarray(ids[lo:hi]),dtype=torch.long,device=device)
        m=torch.as_tensor(np.asarray(mask[lo:hi]),dtype=torch.long,device=device)
        amp=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
        with amp: logits=model(input_ids=x,attention_mask=m).logits
        need(logits.shape==(hi-lo,len(weights)) and bool(torch.isfinite(logits).all()),'Invalid training logits.')
        loss=torch.nn.functional.cross_entropy(logits.float(),y[lo:hi],weight=weights,reduction='sum')/denominator
        need(bool(torch.isfinite(loss)), 'Nonfinite training loss.')
        loss.backward();loss_value+=float(loss.detach());microsteps+=1
    # Check all gradient values via norm; additionally inspect named backbone parameters.
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
    backbone=[p for n,p in model.named_parameters() if n.startswith('roberta.') and 'word_embeddings.weight' in n]
    need(backbone and backbone[0].grad is not None and bool(torch.isfinite(backbone[0].grad).all()) and float(backbone[0].grad.abs().sum())>0,'Missing/nonfinite/zero embedding-backbone gradients.')
    lr=float(opt.param_groups[0]['lr']);opt.step();sched.step()
    need(all(bool(torch.isfinite(p).all()) for p in model.parameters()),'Nonfinite weights after update.')
    opt.zero_grad(set_to_none=True)
    if device.startswith('cuda'):torch.cuda.synchronize()
    return {'rows':len(y),'microsteps':microsteps,'loss':loss_value,'gradient_norm_before_clip':float(norm),
            'learning_rate_used':lr,'seconds':time.monotonic()-start,'backbone_gradients_checked':True}

def forward_probe(model,ids,mask,microbatch,precision,device):
    import torch
    import numpy as np
    model.eval();values=[]
    if device.startswith('cuda'):torch.cuda.synchronize()
    start=time.monotonic()
    with torch.inference_mode():
        for lo in range(0,len(ids),microbatch):
            x=torch.as_tensor(np.asarray(ids[lo:lo+microbatch]),dtype=torch.long,device=device)
            m=torch.as_tensor(np.asarray(mask[lo:lo+microbatch]),dtype=torch.long,device=device)
            amp=torch.autocast('cuda',dtype=torch.bfloat16) if precision=='bf16' else nullcontext()
            with amp:logits=model(input_ids=x,attention_mask=m).logits
            need(bool(torch.isfinite(logits).all()),'Nonfinite validation forward probe.')
            values.append(logits.float().cpu().numpy())
    if device.startswith('cuda'):torch.cuda.synchronize()
    return np.concatenate(values),time.monotonic()-start

def optimizer(model,torch):
    # Exact engineering settings, not a validation-selected scientific result.
    opt=torch.optim.AdamW(model.parameters(),lr=1e-5,betas=(.9,.999),eps=1e-8,weight_decay=.01,foreach=False,fused=False)
    # Three small pilot updates use a 20-step artificial horizon (warmup 2).
    # Full jobs will use their own proposed three-epoch horizon/warmup10%.
    def scale(step):return step/2 if step<2 else max(0.,(20-step)/18)
    sched=torch.optim.lr_scheduler.LambdaLR(opt,scale)
    return opt,sched

def run_session(model_factory,data,folder,signature,mode,microbatch,precision,device):
    """Common CPU-test/CUDA pilot engine; fresh process replay starts at committed step1."""
    import torch
    import numpy as np
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    model=model_factory();opt,sched=optimizer(model,torch);diagnostics=[];step=0;cursor=0
    def restore(name):
        nonlocal step,cursor,diagnostics
        state,pointer=load_state(folder,name,signature)
        model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        step,cursor,diagnostics=state['step'],state['cursor'],state['diagnostics']
        restore_rng(torch,state['rng']);del state
        return pointer
    if mode=='replay':restore('step_1')
    else:
        existing=[k for k in (1,2,3) if (folder/f'step_{k}.ready.json').exists()]
        if existing:restore(f'step_{max(existing)}')
    checkpoint_write=None
    while cursor<len(data['labels']):
        end=min(cursor+32,len(data['labels']))
        d=training_step(model,opt,sched,data['input_ids'][cursor:end],data['attention_mask'][cursor:end],
            data['labels'][cursor:end],data['class_weights'],microbatch,precision,device)
        step+=1;cursor=end;diagnostics.append(d)
        # Always keep step1 for independent replay; further checkpoints only on stop.
        if mode=='reference' and (step==1 or _STOP):
            checkpoint_write=save_state(folder,f'step_{step}',state_pack(torch,model,opt,sched,signature,step,cursor,diagnostics),signature)
        if _STOP:raise PilotStopped('Stopped at optimizer boundary; committed checkpoint retained.')
    need(step==3 and cursor==81,'Unexpected pilot step/cursor coverage.')
    end_state=state_pack(torch,model,opt,sched,signature,step,cursor,diagnostics)
    hashes={k:tree_sha(end_state[k]) for k in ('model','optimizer','scheduler','rng')}
    logits,seconds=forward_probe(model,data['validation_input_ids'],data['validation_attention_mask'],microbatch,precision,device)
    logits_sha=hashlib.sha256(logits.tobytes()).hexdigest()
    record={'state':'PASS','mode':mode,'step':step,'cursor':cursor,'hashes':hashes,'validation_logits_sha256':logits_sha,
            'validation_rows':len(logits),'validation_forward_seconds':seconds,'optimizer_updates':diagnostics,
            'checkpoint_write':checkpoint_write,'test_predictions_created':False,'paper_score_created':False,
            'precision':precision,'microbatch':microbatch,'effective_batch':32,'last_effective_batch':17}
    if device.startswith('cuda'):
        record.update(peak_allocated_mib=torch.cuda.max_memory_allocated(0)/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved(0)/2**20)
    if mode=='reference':write_json(folder/'reference_report.json',record)
    else:
        expected=read_json(folder/'reference_report.json')
        need(expected['step']==step and expected['cursor']==cursor, 'Replay progress differs.')
        for k in hashes:need(expected['hashes'][k]==hashes[k],f'Same-device replay differs in {k}; no silent tolerance relaxation.')
        need(expected['validation_logits_sha256']==logits_sha,'Replay logits differ.')
        record['same_device_exact_replay']=True
        write_json(folder/'replay_report.json',record)
    del model,opt,sched;gc.collect()
    if device.startswith('cuda'):torch.cuda.empty_cache()
    return record


def child(plan_path,mode):
    plan=read_json(plan_path);root=Path(plan_path).parent;device='cuda:0'
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(plan['threads'])
    signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    status=root/(mode+'_status.json');started=time.monotonic()
    try:
        import numpy as np
        import torch
        from transformers import AutoModelForSequenceClassification
        need(torch.cuda.is_available() and torch.cuda.device_count()==1,'Pilot child must see exactly one CUDA device.')
        torch.cuda.set_device(0);torch.set_num_threads(plan['threads'])
        actual=str(getattr(torch.cuda.get_device_properties(0),'uuid',''))
        need(actual.lower().removeprefix('gpu-')==plan['gpu']['uuid'].lower().removeprefix('gpu-'),'Physical CUDA UUID differs from requested GPU.')
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        if plan['precision']=='bf16':need(torch.cuda.is_bf16_supported(),'BF16 is not supported; no silent precision fallback.')
        random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
        for alias in ('features','smoke'):load_helper(plan['project'],alias)
        F=load_helper(plan['project'],'features')
        stages={s['stage']:s for s in read_json(Path(plan['smoke'])/'model_smoke_report.json')['stages']}
        snapshot,asset=F.verify_asset(stages['xlmr'],F.hub_root(plan.get('hub_cache')),'xlmr_tokens')
        need(asset['file_manifest_sha256']==plan['asset_sha256'],'XLM pinned asset changed.')
        datafile=root/'pilot_data.PRIVATE.npz'
        need(sha(datafile)==plan['data']['data_sha256'],'Pilot data checksum differs.')
        with np.load(datafile,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
        def factory():
            m=AutoModelForSequenceClassification.from_pretrained(str(snapshot),num_labels=plan['data']['num_labels'],
                local_files_only=True,trust_remote_code=False,attn_implementation='eager')
            m.float().to(device)
            need(all(p.requires_grad for p in m.parameters()),'Unexpected frozen parameter; pilot is full fine-tuning.')
            need(m.config.model_type=='xlm-roberta','Unexpected base architecture.')
            return m
        torch.cuda.reset_peak_memory_stats(0)
        r=run_session(factory,data,root,plan['signature'],mode,plan['microbatch'],plan['precision'],device)
        r.update(elapsed_seconds=time.monotonic()-started,actual_gpu_uuid=actual,compute_capability=list(torch.cuda.get_device_capability(0)),
                 torch_cuda=torch.version.cuda)
        write_json(root/(mode+'_report.json'),r)
        files=[mode+'_report.json']
        if mode=='reference':
            files+=['step_1.ready.json','step_1.PRIVATE.pt']
        # Phase marker refers to files in parent directory, without unsafe ../ paths.
        write_json(root/(mode+'_complete.json'),{'signature':plan['signature'],'files':{n:{'bytes':(root/n).stat().st_size,'sha256':sha(root/n)} for n in files}})
        write_json(status,{'state':'PASS','elapsed_seconds':time.monotonic()-started})
        return 0
    except BaseException as exc:
        import_name=type(exc).__name__
        oom='OutOfMemory' in import_name
        state='INTERRUPTED' if isinstance(exc,(PilotStopped,KeyboardInterrupt)) else ('OOM_BLOCKED' if oom else 'PILOT_BLOCKED')
        write_json(status,{'state':state,'error_type':import_name,'message':str(exc) if isinstance(exc,PilotError) else 'Local failure; private traceback retained.', 'elapsed_seconds':time.monotonic()-started})
        atomic(root/(mode+'.PRIVATE.log'),lambda f:f.write(traceback.format_exc()),False)
        return 130 if state=='INTERRUPTED' else 1

def phase_done(root,mode,signature):
    path=Path(root)/(mode+'_complete.json')
    if not path.exists():return False
    d=read_json(path);need(d['signature']==signature and d['files'],'Pilot completion identity mismatch.')
    for name,v in d['files'].items():
        f=safe(root,name);need(f.is_file() and f.stat().st_size==v['bytes'] and sha(f)==v['sha256'],'Completed pilot phase is corrupt.')
    return True

def launch(plan,mode,gpu):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1';env['TOKENIZERS_PARALLELISM']='false'
    private=Path(plan).parent/(mode+'_console.PRIVATE.log')
    with private.open('a',encoding='utf-8') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child',mode,'--plan',str(plan)],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        signaled=False
        while p.poll() is None:
            if _STOP and not signaled:p.send_signal(signal.SIGTERM);signaled=True
            time.sleep(.25)
    need(p.returncode==0,'Pilot child did not pass; read the shareable stage status, not private logs first.')

def package(root,report):
    write_json(root/'xlmr_pilot_report.json',report)
    lines=['SAUCIS — XLM-R ETKIN BATCH / CHECKPOINT PILOTU','='*64,'Durum: '+report['state'],
        'Protocol: '+report.get('protocol_id','unknown'),
        'Bu pilot tam egitim/hiperparametre secimi/final-test degerlendirmesi DEGIL.',
        'Her scope: 3 optimizer adimi (32+32+17 kayit); ayni GPUda ayri process replay.',
        'Tam XLM-R backbone, sabit256padding; yeni egitim parametreleri otomatik kilitlenmedi.']
    for s in report.get('stages',[]):lines.append(f"{s['context']} / {s['precision']}: {s['state']}")
    if report.get('message'):lines.append('Bilgi: '+report['message'])
    lines.append('PAYLAS: shareable_xlmr_pilot.zip. PRIVATE checkpoint/ID/log dosyalari yerelde kalsin.')
    atomic(root/'xlmr_pilot_summary.txt',lambda f:f.write('\n'.join(lines)+'\n'),False)
    names=['xlmr_pilot_report.json','xlmr_pilot_summary.txt','linear_audit.json','future_xlmr_jobs.json','training_settings_proposal.json']
    def writer(f):
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            for n in names:
                if (root/n).is_file():z.write(root/n,n)
    atomic(root/'shareable_xlmr_pilot.zip',writer)
    print('\n'.join(lines),flush=True);print('[SHARE]',root/'shareable_xlmr_pilot.zip',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.');p.add_argument('--split');p.add_argument('--features');p.add_argument('--smoke');p.add_argument('--linear')
    p.add_argument('--gpu',default='1',help='Explicit physical nvidia-smi index or full GPU UUID.')
    p.add_argument('--microbatch',type=int,default=4);p.add_argument('--precisions',nargs='+',choices=['fp32','bf16'],default=['fp32','bf16'])
    p.add_argument('--threads',type=int,default=4);p.add_argument('--hub-cache')
    p.add_argument('--child',choices=['reference','replay'],help=argparse.SUPPRESS);p.add_argument('--plan',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.child:return child(args.plan,args.child)
    need(all((args.split,args.features,args.smoke,args.linear)),'Provide split/features/smoke/linear paths.')
    need(args.microbatch in (1,2,4,8,16,32),'Microbatch must divide effective batch32.')
    need(len(args.precisions)==len(set(args.precisions)) and args.threads>0,'Duplicate precision/invalid threads.')
    project=Path(args.project).expanduser().resolve()
    def resolve(v):
        q=Path(v).expanduser();return (q if q.is_absolute() else project/q).resolve()
    split,features,smoke,linear=map(resolve,(args.split,args.features,args.smoke,args.linear))
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]=str(args.threads)
    L=load_helper(project,'linear');F=load_helper(project,'features');S=load_helper(project,'smoke')
    base,fr,contexts=L.inspect_inputs(split,features,smoke)
    # Enforce neural/tokenizer package identity in addition to the linear checks.
    fbase,stages=F.inspect_inputs(split,smoke)
    audited=audit_linear_directory(linear,base)
    entry=fr['banks']['xlmr_tokens'];bank=F.FeatureBank(L.safe_path(features,entry['bank_directory']))
    need(bank.spec['protocol_id']==base['protocol_id'] and bank.spec['corpus_sha256']==base['corpus_sha256'] and bank.spec['revision']==stages['xlmr']['asset']['revision'], 'Token bank/pinned model/corpus mismatch.')
    need(digest(canonical(bank.spec))==entry['spec_sha256'] and bank.spec['asset_sha256']==stages['xlmr']['asset']['file_manifest_sha256'],'XLM token spec/asset mismatch.')
    snapshot,asset=F.verify_asset(stages['xlmr'],F.hub_root(args.hub_cache),'xlmr_tokens')
    cfg=read_json(snapshot/'config.json');pad=cfg.get('pad_token_id');need(isinstance(pad,int),'Pinned pad token missing.')
    gpu=S.safe_gpu_info(args.gpu);code=sha(__file__)
    identity={'protocol_id':base['protocol_id'],'corpus_sha256':base['corpus_sha256'],'code_sha256':code,
              'split_report_sha256':base['split_report_sha256'],'feature_report_sha256':base['feature_report_sha256'],
              'smoke_report_sha256':base['smoke_report_sha256'],'linear_report_sha256':audited['report_sha256'],
              'asset_sha256':asset['file_manifest_sha256'],'gpu_uuid':gpu['uuid'],'microbatch':args.microbatch,
              'precisions':args.precisions,'packages':fbase['packages'],'threads':args.threads}
    root=project/'_revision_xlmr_pilot'/('pilot_'+base['protocol_id']+'_'+digest(canonical(identity))[:10]);root.mkdir(parents=True,exist_ok=True)
    print('[OUTPUT]',root,flush=True);print('[GPU]',gpu['index'],gpu['uuid'],gpu['name'],flush=True)
    report={'state':'RUNNING','created_at':now(),'protocol_id':base['protocol_id'],'identity':identity,'gpu':gpu,
        'linear_audit':audited,'stages':[],'final_test_predictions_created':False,'paper_scores_created':False,
        'production_hyperparameters_selected':False,'limits':['Pilot only; not full training or a benchmark result.','Same-GPU process replay only; cross-GPU replay is not tested.','No precision fallback or automatic microbatch reduction.']}
    rc=0;signal.signal(signal.SIGTERM,stop_handler);signal.signal(signal.SIGINT,stop_handler)
    with L.lock(root/'.lock'):
        try:
            if (root/'identity.json').exists():need(read_json(root/'identity.json')==identity,'Pilot identity changed.')
            else:write_json(root/'identity.json',identity)
            write_json(root/'linear_audit.json',audited)
            jobs=future_jobs(L,split,contexts)
            write_json(root/'future_xlmr_jobs.json',{'count':len(jobs),'jobs':jobs,'executed_here':False})
            write_json(root/'training_settings_proposal.json',{
                'status':'proposal_for_full_runner_review_not_executed_here','model':'FacebookAI/xlm-roberta-base','revision':asset['revision'],
                'families':list(APPROVED_FAMILIES),'seeds':[42,43,44,45,46],'learning_rates':[1e-5,2e-5],'max_epochs':3,'effective_batch':32,'max_length':256,
                'optimizer':{'name':'AdamW','betas':[.9,.999],'eps':1e-8,'weight_decay':.01,'weight_decay_groups':'all trainable parameters, matching pilot','foreach':False,'fused':False},
                'scheduler':'linear decay, warmup ceil(0.1*total_optimizer_steps)','max_grad_norm':1.,
                'loss':'training-frequency balanced cross-entropy, one weight denominator for each effective batch',
                'precision':'TO_BE_LOCKED_AFTER_ENGINEERING_PILOT','microbatch':'TO_BE_LOCKED_AFTER_ENGINEERING_PILOT',
                'selection':'source-only validation; joint equal-language mean; LR/epoch selected within each training context',
                'pilot_difference':'pilot uses artificial20step horizon,warmup2 and3updates; not full epochs or tuning'})
            for context_id in PILOT_CONTEXTS:
                context=contexts[42][context_id]
                for precision in args.precisions:
                    if _STOP:raise PilotStopped('Stop requested between pilot scopes.')
                    scope=root/(L.safe_name(context_id)+'_'+precision);scope.mkdir(parents=True,exist_ok=True)
                    d=prepare_data(L,F,split,features,fr,scope,context,pad)
                    plan={'project':str(project),'smoke':str(smoke),'hub_cache':str(F.hub_root(args.hub_cache)),
                          'data':d,'gpu':gpu,'asset_sha256':asset['file_manifest_sha256'],'precision':precision,
                          'microbatch':args.microbatch,'threads':args.threads,'pilot_identity':identity}
                    plan['signature']=digest(canonical({k:v for k,v in plan.items() if k not in ('project','smoke','hub_cache','gpu')}|{'gpu_uuid':gpu['uuid']}))
                    planfile=scope/'pilot_plan.local.json'
                    if planfile.exists():
                        old=read_json(planfile);need(old['signature']==plan['signature'],'Scope input identity changed.')
                    write_json(planfile,plan)
                    s={'context':context_id,'precision':precision,'state':'RUNNING','data':d}
                    report['stages'].append(s)
                    for mode in ('reference','replay'):
                        if phase_done(scope,mode,plan['signature']):print('[REUSE]',context_id,precision,mode,flush=True)
                        else:
                            print('[PILOT]',context_id,precision,mode,'effective_batch=32 microbatch='+str(args.microbatch),flush=True)
                            try:launch(planfile,mode,gpu)
                            except BaseException:
                                if (scope/(mode+'_status.json')).exists():s['failure']=read_json(scope/(mode+'_status.json'))
                                s['state']='BLOCKED';raise
                        need(phase_done(scope,mode,plan['signature']),'Child returned without valid completion marker.')
                        s[mode]=read_json(scope/(mode+'_report.json'))
                    s['state']='PASS';print('[PASS]',context_id,precision,flush=True)
            report['state']='XLMR_PILOT_PASSED'
        except BaseException as exc:
            report['state']='INTERRUPTED' if _STOP or isinstance(exc,(PilotStopped,KeyboardInterrupt)) else 'XLMR_PILOT_BLOCKED'
            report['error_type']=type(exc).__name__;report['message']=str(exc) if isinstance(exc,(PilotError,L.TrainingError,F.FeatureError)) else 'Local error; PRIVATE log retained.'
            atomic(root/'error.PRIVATE.log',lambda f:f.write(traceback.format_exc()),False);rc=130 if report['state']=='INTERRUPTED' else 1
        finally:
            report['finished_at']=now();package(root,report)
    return rc

if __name__=='__main__':
    try:sys.exit(main())
    except PilotError as exc:print('[BLOCKED]',str(exc),file=sys.stderr);sys.exit(1)
