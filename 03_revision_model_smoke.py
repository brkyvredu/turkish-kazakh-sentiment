#!/usr/bin/env python3
"""SAUCIS revision: engineering smoke tests on immutable step-02 manifests.

No final-test predictions or paper scores are produced. No source, split or legacy
result is overwritten. Only small train/validation subsets are fitted. Models run
one at a time in separate processes on an explicitly selected GPU. XLM-R download
requires --allow-xlmr-download; no pip installation is ever performed.
Python >=3.10. CPU: numpy, scipy, scikit-learn, joblib. GPU: torch,
transformers, sentence-transformers, huggingface-hub (already in the project).
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
import gc
import hashlib
import importlib.metadata as md
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import sqlite3
import subprocess
import sys
import time
import zipfile

VERSION = '1.0.0'
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
FROZEN = {
 'minilm': ('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2',
            'e8f8c211226b894fcb81acc59f3b34ba3efd5f42', 384, 128),
 'labse': ('sentence-transformers/LaBSE', '836121a0533e5664b21c7aacc5d22951f2b8b25b', 768, 256),
 'bge_m3': ('BAAI/bge-m3', '5617a9f61b028005a4858fdac845db406aefb181', 1024, 8192),
}
PACKAGES = ('numpy','scipy','scikit-learn','joblib','torch','transformers',
            'sentence-transformers','huggingface-hub','accelerate','safetensors','tokenizers')

class SmokeError(RuntimeError):
    """Messages intentionally do not include review text or original record IDs."""

class MissingAsset(SmokeError): pass

def require(condition, message):
    if not condition: raise SmokeError(message)

def now(): return datetime.now(timezone.utc).isoformat()
def canonical(x): return json.dumps(x, ensure_ascii=False, sort_keys=True, separators=(',',':'), allow_nan=False)
def digest(x): return hashlib.sha256(x.encode('utf-8')).hexdigest()
def file_sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read_json(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def write_json(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+'.partial_'+str(time.time_ns()))
    with temp.open('w',encoding='utf-8') as f:
        json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p)
def versions():
    result={'python':platform.python_version()}
    for k in PACKAGES:
        try: result[k]=md.version(k)
        except md.PackageNotFoundError: result[k]=None
    return result

@contextmanager
def locked(path):
    import fcntl
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    with Path(path).open('a') as f:
        try: fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise SmokeError('This smoke output is already in use.') from e
        try: yield
        finally: fcntl.flock(f,fcntl.LOCK_UN)

@contextmanager
def ro_db(p):
    p=Path(p).resolve();require(p.is_file(),'Required local SQLite file is missing.')
    db=sqlite3.connect(p.as_uri()+'?mode=ro',uri=True)
    db.execute('PRAGMA query_only=ON')
    try: yield db
    finally: db.close()

@dataclass(frozen=True)
class Sample:
    sample_id: str
    group_id: str
    text: str
    target: int
    language: str


def pick_dev(split:Path, seed:int, set_id:str, per_class:int):
    """Read ONLY a training/validation set. Deterministic per-language/class sample."""
    require(per_class>0,'Positive smoke sample size required.')
    with ro_db(split/f'seed_{seed}'/'memberships.local.sqlite3') as db:
        meta=db.execute('SELECT role,task FROM sets WHERE set_id=?',(set_id,)).fetchone()
        require(meta is not None,'Unknown subset.')
        require(meta[0] in ('train','validation'),'Smoke reader refuses final-test data.')
        db.execute('ATTACH DATABASE ? AS corpus',((split/'corpus.local.sqlite3').resolve().as_uri()+'?mode=ro',))
        buckets=defaultdict(list)
        for sid,gid,txt,target,lang in db.execute('''SELECT r.sample_id,r.group_id,r.model_text,m.target,r.language
          FROM members m JOIN corpus.records r USING(row_id) WHERE m.set_id=? ORDER BY r.row_id''',(set_id,)):
            buckets[(lang,target)].append(Sample(sid,gid,txt,int(target),lang))
    require(buckets,'Empty development set.')
    chosen=[]
    for key,rows in sorted(buckets.items()):
        require(len(rows)>=per_class,'A development smoke quota is unavailable; no silent clipping.')
        chosen.extend(sorted(rows,key=lambda r:digest(canonical(['smoke-v1',seed,set_id,r.sample_id])))[:per_class])
    return chosen,meta[1]


def validate_development(a,b,task):
    require({r.group_id for r in a}.isdisjoint({r.group_id for r in b}),'Development groups overlap.')
    labs={0,1} if task=='binary' else {0,1,2}
    require({r.target for r in a}==labs and {r.target for r in b}==labs,'Unexpected task classes.')
    require({r.language for r in a}=={r.language for r in b},'Validation language differs from source training language.')


def inspect_manifests(split):
    """Verify complete-file checksums and reported checks, without training."""
    split=Path(split);report=read_json(split/'split_report.json');protocol=read_json(split/'protocol.json')
    require(report['state']=='SPLIT_MANIFESTS_READY' and report['blocked_checks']==0,'Split gate is not ready.')
    require(report['protocol_id']==protocol['protocol_id'],'Protocol identity mismatch.')
    require(read_json(split/'status.json')['report_sha256']==file_sha(split/'split_report.json'),'Split report checksum mismatch.')
    corpus_stamp=read_json(split/'corpus.complete.json')
    require(corpus_stamp['sha256']==file_sha(split/'corpus.local.sqlite3'),'Corpus checksum mismatch.')
    with ro_db(split/'corpus.local.sqlite3') as db:
        require(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corpus integrity failure.')
    per_seed={};observed={};total=0
    for seed in report['seeds']:
        root=split/f'seed_{seed}';done=read_json(root/'complete.json')
        for name,key in [('memberships.local.sqlite3','db_sha256'),('seed_report.json','report_sha256'),('experiments.json','recipes_sha256')]:
            require(file_sha(root/name)==done[key],f'Seed {seed}: completed file checksum mismatch.')
        for name,h in done.get('table_checksums',{}).items():
            require(file_sha(root/name)==h,f'Seed {seed}: diagnostic checksum mismatch.')
        r=read_json(root/'seed_report.json');recipes=read_json(root/'experiments.json')
        require(r['state']=='SPLIT_MANIFESTS_READY' and not r['errors'],'A seed is not ready.')
        require(all(c['status']=='PASS' for c in r['checks']),'Unpassed split checks found.')
        require(len({s['set_id'] for s in r['sets']})==len(r['sets']),'Duplicate set names.')
        total+=len(r['checks']);sets={s['set_id']:s for s in r['sets']};contexts={}
        for e in recipes:
            a,b,c=(sets[e[k]] for k in ('train_set','validation_set','test_set'))
            require((a['role'],b['role'],c['role'])==('train','validation','test'),'Incorrect recipe roles.')
            require(a['task']==b['task']==c['task'],'Mixed task schemas.')
            require(set(b['language_counts'])==set(e['validation_languages']),'Wrong selection languages.')
            require(e['test_labels_must_not_select_models'] is True,'Unsafe model selection recipe.')
            signature=(e['train_set'],e['validation_set'],e['selection_metric'],tuple(e['validation_languages']))
            ctx=e['training_context_id']
            if ctx in contexts: require(contexts[ctx]==signature,'A shared training context has inconsistent selection data.')
            contexts[ctx]=signature
        per_seed[str(seed)]={'subsets':len(sets),'evaluation_recipes':len(recipes),'training_contexts':len(contexts)}
        for name,s in sets.items():observed.setdefault(name,[]).append(s['manifest_sha256'])
    require(total==report['checks_count'],'Split check total mismatch.')
    distinct={k:len(set(v)) for k,v in observed.items() if k.startswith('main:')}
    require(all(n==len(report['seeds']) for n in distinct.values()),'Main subsets are not distinct across all seeds.')
    return {'protocol_id':report['protocol_id'],'verified_split_checks':total,'per_seed':per_seed,
            'distinct_main_subset_hashes':distinct,'final_test_predictions_created':False}


def metric_pack(y,p,labels):
    import numpy as np
    from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix
    y=np.asarray(y);p=np.asarray(p)
    require(len(y)>0 and len(y)==len(p),'Invalid metric vector lengths.')
    require(set(y)<=set(labels) and set(p)<=set(labels),'Metric class outside task schema.')
    return {'accuracy':float(accuracy_score(y,p)),
      'macro_f1':float(f1_score(y,p,labels=labels,average='macro',zero_division=0)),
      'weighted_f1':float(f1_score(y,p,labels=labels,average='weighted',zero_division=0)),
      'class_report':classification_report(y,p,labels=labels,output_dict=True,zero_division=0),
      'confusion_matrix':confusion_matrix(y,p,labels=labels).tolist()}


def selection_score(y,p,langs,labels,joint):
    import numpy as np
    langs=np.asarray(langs);y=np.asarray(y);p=np.asarray(p)
    if not joint: return metric_pack(y,p,labels)['macro_f1']
    require(set(langs)=={'tr','kz'},'Joint selection must evaluate both languages.')
    return sum(metric_pack(y[langs==k],p[langs==k],labels)['macro_f1'] for k in ('tr','kz'))/2


def make_classifier(kind,C,seed=42):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    if kind=='lr': return LogisticRegression(C=C,solver='lbfgs',max_iter=3000,class_weight='balanced',random_state=seed)
    if kind=='svm': return LinearSVC(C=C,dual='auto',max_iter=10000,class_weight='balanced',random_state=seed)
    raise SmokeError('Unknown classifier.')


def fit_select(Xtr,ytr,Xval,yval,langs,labels,kind,seed,joint=False):
    """Deliberately no final-test argument. Validation selects C; ties choose smaller C."""
    import warnings
    from sklearn.exceptions import ConvergenceWarning
    trials=[];winner=None;score=-math.inf
    for C in C_GRID:
        model=make_classifier(kind,C,seed)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(Xtr,ytr)
        require(not any(issubclass(w.category,ConvergenceWarning) for w in caught),'Classifier did not converge; do not silently accept this fit.')
        value=selection_score(yval,model.predict(Xval),langs,labels,joint)
        require(math.isfinite(value),'Nonfinite validation score.')
        trials.append({'C':C,'validation_score':value})
        if value>score: score=value;winner=model
    require(winner is not None,'Model selection failed.')
    return winner,trials


def vectorizer(kind):
    from sklearn.feature_extraction.text import TfidfVectorizer
    require(kind in ('word','char'),'Unknown TF-IDF representation.')
    return TfidfVectorizer(analyzer=kind,ngram_range=(1,2) if kind=='word' else (3,5),
        max_features=80000 if kind=='word' else 120000,min_df=2,lowercase=True,
        sublinear_tf=True,smooth_idf=True,norm='l2',strip_accents=None)


def save_checkpoint(path,state):
    import torch
    p=Path(path);tmp=p.with_name(p.name+'.partial_'+str(time.time_ns()))
    with tmp.open('wb') as f:
        torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    write_json(p.with_name(p.name+'.json'),{'sha256':file_sha(p)})


def load_checkpoint(path):
    import torch
    p=Path(path);h=read_json(p.with_name(p.name+'.json'))['sha256']
    require(file_sha(p)==h,'Own checkpoint checksum mismatch.')
    return torch.load(p,map_location='cpu',weights_only=True)


def cpu_tree(x):
    import torch
    if isinstance(x,torch.Tensor):return x.detach().cpu().clone()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    if isinstance(x,tuple):return tuple(cpu_tree(v) for v in x)
    return x


def toy_resume_test(folder):
    """Real CPU PyTorch dropout/AdamW/scheduler resume, not an XLM-R test."""
    import torch
    torch.manual_seed(42)
    def new():
        m=torch.nn.Sequential(torch.nn.Linear(5,8),torch.nn.Dropout(.25),torch.nn.Linear(8,3))
        o=torch.optim.AdamW(m.parameters(),lr=1e-3);s=torch.optim.lr_scheduler.LambdaLR(o,lambda step:1/(step+1))
        return m,o,s
    model,opt,sched=new();x=torch.randn(6,5);y=torch.tensor([0,1,2,0,1,2])
    def step(m,o,s):
        m.train();o.zero_grad(set_to_none=True);loss=torch.nn.functional.cross_entropy(m(x),y)
        loss.backward();o.step();s.step();return float(loss.detach())
    step(model,opt,sched)
    p=Path(folder)/'toy_checkpoint.pt'
    save_checkpoint(p,{'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),
                        'scheduler':sched.state_dict(),'rng':torch.get_rng_state(),'step':1})
    expected=step(model,opt,sched);final=cpu_tree(model.state_dict())
    other,oo,ss=new();state=load_checkpoint(p);other.load_state_dict(state['model']);oo.load_state_dict(state['optimizer'])
    ss.load_state_dict(state['scheduler']);torch.set_rng_state(state['rng']);actual=step(other,oo,ss)
    require(expected==actual and all(torch.equal(final[k],v) for k,v in other.state_dict().items()),'CPU checkpoint continuation changed results.')
    require(sched.state_dict()==ss.state_dict(),'CPU scheduler restoration failed.')
    return {'passed':True,'model':'tiny_synthetic_torch_network','bitwise_same_on_this_CPU_run':True}


def cache_fingerprint(samples,config):
    # Includes both ordering and text bytes, not merely row count or sample IDs.
    return digest(canonical({'config':config,'records':[(r.sample_id,digest(r.text)) for r in samples]}))


def write_embedding_cache(path,emb,samples,config):
    import numpy as np
    p=Path(path);tmp=p.with_name(p.name+'.partial_'+str(time.time_ns()))
    arr=np.asarray(emb,dtype=np.float32)
    require(arr.ndim==2 and len(arr)==len(samples) and np.isfinite(arr).all(),'Invalid embedding matrix.')
    with tmp.open('wb') as f:np.save(f,arr,allow_pickle=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
    write_json(p.with_suffix('.json'),{'fingerprint':cache_fingerprint(samples,config),'shape':list(arr.shape),'dtype':str(arr.dtype),'sha256':file_sha(p)})


def read_embedding_cache(path,samples,config):
    import numpy as np
    p=Path(path);meta=read_json(p.with_suffix('.json'))
    require(meta['fingerprint']==cache_fingerprint(samples,config),'Embedding cache order/text/config mismatch.')
    require(meta['sha256']==file_sha(p),'Embedding file checksum mismatch.')
    arr=np.load(p,allow_pickle=False)
    require(arr.ndim==2 and list(arr.shape)==meta['shape'] and len(arr)==len(samples),'Embedding shape mismatch.')
    require(str(arr.dtype)==meta['dtype'] and np.isfinite(arr).all(),'Embedding dtype/values invalid.')
    return arr


def cpu_stage(split,out):
    import numpy as np
    import joblib
    import inspect
    out=Path(out);checks=[];fit_count=0;contexts=[]
    # Test all recipe contexts in seed 42 on tiny development-only subsets.
    refused=False
    try: pick_dev(Path(split),42,'main:tr:test',1)
    except SmokeError as exc: refused='refuses final-test' in str(exc)
    require(refused,'Smoke data loader did not refuse the final-test set.')
    recipes=read_json(Path(split)/'seed_42'/'experiments.json');by_context={}
    for e in recipes:by_context.setdefault(e['training_context_id'],e)
    require('test' not in inspect.signature(fit_select).parameters,'Unexpected test parameter.')
    details=[]
    for ctx,e in by_context.items():
        print('[CPU CONTEXT]',ctx,flush=True)
        a,task=pick_dev(Path(split),42,e['train_set'],24);b,task2=pick_dev(Path(split),42,e['validation_set'],8)
        require(task==task2,'Task changed.');validate_development(a,b,task)
        labels=[0,1] if task=='binary' else [0,1,2]
        y=np.array([r.target for r in a]);v=np.array([r.target for r in b]);langs=[r.language for r in b]
        for kind in ('word','char'):
            tf=vectorizer(kind);X=tf.fit_transform([r.text for r in a]);vocab=dict(tf.vocabulary_);idf=tf.idf_.copy()
            Y=tf.transform([r.text for r in b]);tf.transform(['zzzzsmokevalidationonlytokenzzzz'])
            require(tf.vocabulary_==vocab and np.array_equal(tf.idf_,idf),'Validation transform modified TF-IDF fit state.')
            require('zzzzsmokevalidationonlytokenzzzz' not in tf.vocabulary_,'Validation sentinel entered vocabulary.')
            for clf in ('lr','svm'):
                m,trials=fit_select(X,y,Y,v,langs,labels,clf,42,e['selection_metric']=='equal_language_mean_macro_f1')
                pred=m.predict(Y);require(np.isfinite(m.decision_function(Y)).all(),'Invalid decision scores.')
                metric_pack(v,pred,labels);fit_count+=len(trials)
                dest=out/(digest(ctx+kind+clf)[:16]+'.PRIVATE.joblib')
                joblib.dump({'vectorizer':tf,'classifier':m},dest)
                reloaded=joblib.load(dest)
                require(np.array_equal(pred,reloaded['classifier'].predict(reloaded['vectorizer'].transform([r.text for r in b]))),'Classifier roundtrip changed predictions.')
                details.append({'context':ctx,'representation':kind,'classifier':clf,'selected_C':m.C,'trials':trials})
        contexts.append({'context':ctx,'task':task,'train_rows':len(a),'validation_rows':len(b),'selection_languages':sorted(set(langs))})
    # Standalone metric ground truth (binary F1 and zero-F1 for an absent prediction class).
    require(abs(metric_pack([0,0,1,1],[0,1,1,1],[0,1])['macro_f1']-(2/3+4/5)/2)<1e-12,'Binary metric test failed.')
    require(abs(metric_pack([0,1,2],[0,0,0],[0,1,2])['macro_f1']-1/6)<1e-12,'Three-class metric test failed.')
    samples=[Sample('a','ga','Ürün iyi',0,'tr'),Sample('b','gb','Пікір басқа',1,'kz')]
    cfg={'model':'synthetic','revision':'a','normalize':True,'protocol':'smoke'}
    cp=out/'synthetic_cache.npy';arr=np.array([[1.,0.],[0.,1.]],dtype=np.float32)
    write_embedding_cache(cp,arr,samples,cfg);require(np.array_equal(read_embedding_cache(cp,samples,cfg),arr),'Cache roundtrip failed.')
    rejected=0
    for bad_samples,bad_cfg in [(samples[::-1],cfg),([samples[0],Sample('b','gb','changed text',1,'kz')],cfg),(samples,{**cfg,'revision':'b'})]:
        try:read_embedding_cache(cp,bad_samples,bad_cfg)
        except SmokeError:rejected+=1
    require(rejected==3,'Invalid cache reuse was not rejected.')
    write_json(out/'pilot_trials.PRIVATE.json',details)
    toy=toy_resume_test(out)
    return {'state':'PASS','contexts':contexts,'source_validation_only':True,'final_test_reader_refused':True,
            'candidate_fits':fit_count,'pipeline_combinations':len(details),'fit_scope':'small train/validation only',
            'cache_mismatch_cases_rejected':rejected,'metric_checks':2,'toy_resume':toy,
            'paper_results':False,'note':'Pilot scores remain local diagnostics, not paper comparisons.'}


def safe_gpu_info(selector):
    cmd=['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.free','--format=csv,noheader,nounits']
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
    require(r.returncode==0,'nvidia-smi query failed.')
    cards=[]
    for line in r.stdout.strip().splitlines():
        p=[v.strip() for v in line.split(',')]
        require(len(p)==5,'Unexpected GPU query format.')
        cards.append(dict(index=int(p[0]),uuid=p[1],name=p[2],total_mib=int(p[3]),free_mib=int(p[4])))
    matches=[g for g in cards if str(g['index'])==str(selector) or g['uuid']==selector]
    require(len(matches)==1,'Requested physical GPU index/UUID is unavailable.')
    g=matches[0];mask=os.environ.get('CUDA_VISIBLE_DEVICES')
    if mask is not None:
        tokens=[t.strip() for t in mask.split(',') if t.strip()]
        allowed=str(g['index']) in tokens or any(t.startswith('GPU-') and g['uuid'].startswith(t) for t in tokens)
        require(allowed,'Requested GPU is excluded by inherited CUDA_VISIBLE_DEVICES; mask is not widened.')
    return g


def torch_setup():
    import torch
    torch.set_num_threads(2)
    require(torch.cuda.is_available(),'PyTorch CUDA is not available.')
    require(torch.cuda.device_count()==1,'Child process must see exactly one GPU.')
    torch.cuda.set_device(0);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    return torch


def cuda_probe(out):
    torch=torch_setup();d=torch.device('cuda:0');prop=torch.cuda.get_device_properties(0)
    x=torch.randn(128,128,device=d,requires_grad=True);loss=(x@x.T).square().mean();loss.backward()
    require(torch.isfinite(loss).item() and torch.isfinite(x.grad).all().item(),'CUDA forward/backward failed.')
    bf16=False
    if torch.cuda.is_bf16_supported():
        x.grad=None
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):q=(x@x.T).float().square().mean()
        q.backward();require(torch.isfinite(q).item() and torch.isfinite(x.grad).all().item(),'BF16 CUDA probe failed.');bf16=True
    torch.cuda.synchronize()
    return {'state':'PASS','name':prop.name,'torch_uuid':str(getattr(prop,'uuid','unavailable')),
            'compute_capability':list(torch.cuda.get_device_capability(0)),'torch_cuda':torch.version.cuda,
            'fp32_forward_backward':True,'bf16_probe_passed':bf16,'production_precision_not_selected':True}


def snapshot_manifest(path,alias):
    p=Path(path);require(p.is_dir(),'Pinned model snapshot is missing.')
    require((p/'config.json').is_file(),'Pinned model config is missing.')
    if alias!='xlmr':require((p/'modules.json').is_file(),'Pinned SentenceTransformer module configuration is missing.')
    entries={}
    for q in sorted(p.rglob('*')):
        if q.is_file() and (q.suffix in ('.json','.safetensors','.bin','.model','.txt')):
            require(q.stat().st_size>0,'Empty required snapshot file.')
            entries[str(q.relative_to(p))]={'bytes':q.stat().st_size,'sha256':file_sha(q)}
    require(any(k.endswith(('.bin','.safetensors')) for k in entries),'Pinned snapshot has no model weights.')
    return {'alias':alias,'revision':p.name,'files':entries,'file_manifest_sha256':digest(canonical(entries))}


def frozen_stage(split,out,alias,snapshot,asset):
    import numpy as np
    from sentence_transformers import SentenceTransformer
    torch=torch_setup();started=time.monotonic()
    a,task=pick_dev(Path(split),42,'main:joint:train',24);b,_=pick_dev(Path(split),42,'main:joint:validation',8)
    validate_development(a,b,task);samples=a+b
    model=SentenceTransformer(str(snapshot),device='cuda:0',local_files_only=True,trust_remote_code=False)
    model.eval();model.float()
    require(all(p.device.type=='cuda' and p.device.index==0 for p in model.parameters()),'Encoder escaped selected GPU.')
    require(getattr(model,'default_prompt_name',None) is None,'Unexpected default prompt; configuration needs review.')
    require(model.max_seq_length==FROZEN[alias][3],'Pinned encoder max_seq_length changed.')
    require(model.get_sentence_embedding_dimension()==FROZEN[alias][2],'Encoder dimension differs from audited model.')
    # Text content and ordering are part of each immutable shard identity.
    cfg={'protocol':read_json(Path(split)/'protocol.json')['protocol_id'],'asset':asset['file_manifest_sha256'],
         'revision':asset['revision'],'normalize':True,'dtype':'float32','max_seq_length':model.max_seq_length,
         'prompt':None,'batch_size':4,'implementation':'sentence_transformers.encode'}
    arrays=[];reused=0
    for j,start in enumerate(range(0,len(samples),64)):
        shard=samples[start:start+64];p=Path(out)/f'embeddings_{j:03d}.npy'
        if p.exists() and p.with_suffix('.json').exists():
            arr=read_embedding_cache(p,shard,cfg);reused+=1
        else:
            arr=model.encode([r.text for r in shard],batch_size=4,show_progress_bar=False,
                             convert_to_numpy=True,normalize_embeddings=True)
            write_embedding_cache(p,arr,shard,cfg)
            arr=read_embedding_cache(p,shard,cfg)
        arrays.append(arr)
    E=np.concatenate(arrays);require(np.allclose(np.linalg.norm(E,axis=1),1,atol=2e-4),'Embeddings not L2-normalized.')
    again=model.encode([r.text for r in samples[:4]],batch_size=4,show_progress_bar=False,convert_to_numpy=True,normalize_embeddings=True)
    require(np.allclose(E[:4],again,rtol=2e-4,atol=2e-5),'Encoder repeat check differs beyond smoke tolerance.')
    lengths=[]
    for start in range(0,len(samples),32):
        tokens=model.tokenizer([r.text for r in samples[start:start+32]],padding=False,truncation=False,add_special_tokens=True)
        lengths.extend(len(ids) for ids in tokens['input_ids'])
    y=np.array([r.target for r in a]);yv=np.array([r.target for r in b]);langs=[r.language for r in b]
    trials=[]
    for kind in ('lr','svm'):
        for binary in (False,True):
            ti=y!=1 if binary else np.ones(len(y),bool);vi=yv!=1 if binary else np.ones(len(yv),bool)
            yy=(y[ti]==2).astype(int) if binary else y[ti];vv=(yv[vi]==2).astype(int) if binary else yv[vi]
            labs=[0,1] if binary else [0,1,2]
            clf,ts=fit_select(E[:len(a)][ti],yy,E[len(a):][vi],vv,np.array(langs)[vi],labs,kind,42,True)
            metric_pack(vv,clf.predict(E[len(a):][vi]),labs)
            trials.append({'classifier':kind,'task':'binary' if binary else 'three_class','trials':ts})
    write_json(Path(out)/'pilot_trials.PRIVATE.json',trials)
    torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated(0)
    return {'state':'PASS','alias':alias,'revision':asset['revision'],'dimension':E.shape[1],
      'max_seq_length':model.max_seq_length,'train_rows':len(a),'validation_rows':len(b),
      'shards':len(arrays),'reused_shards':reused,'finite_normalized_embeddings':True,
      'repeat_encode_passed':True,'binary_and_three_class_heads_passed':True,
      'pilot_truncated_rows':sum(n>model.max_seq_length for n in lengths),'pilot_max_tokens':max(lengths),
      'peak_allocated_mib':peak/2**20,'elapsed_seconds':time.monotonic()-started,
      'scope':'small development sample; NOT full-corpus truncation profiling or resource sizing',
      'paper_results':False}


def rng_state(torch):return {'cpu':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(0)}
def restore_rng(torch,state):torch.set_rng_state(state['cpu']);torch.cuda.set_rng_state(state['cuda'],0)


def xlmr_stage(split,out,snapshot,asset):
    from transformers import AutoTokenizer,AutoModelForSequenceClassification
    torch=torch_setup();started=time.monotonic();out=Path(out)
    a,task=pick_dev(Path(split),42,'main:tr:train',24);b,_=pick_dev(Path(split),42,'main:tr:validation',8)
    validate_development(a,b,task)
    tokenizer=AutoTokenizer.from_pretrained(str(snapshot),local_files_only=True,trust_remote_code=False,use_fast=True)
    # Deterministic small batches: smoke uses 2 physical examples, NOT production effective batch 32.
    batch_rows=[(a[0],a[24]),(a[48],a[1])]
    batches=[(dict(tokenizer([r.text for r in rows],padding='max_length',max_length=256,truncation=True,return_tensors='pt')),
              torch.tensor([r.target for r in rows],dtype=torch.long)) for rows in batch_rows]
    batches=[({k:v.to('cuda:0') for k,v in x.items()},y.to('cuda:0')) for x,y in batches]
    signature=digest(canonical({'asset':asset['file_manifest_sha256'],'protocol':read_json(Path(split)/'protocol.json')['protocol_id'],
                         'samples':[(r.sample_id,digest(r.text)) for rows in batch_rows for r in rows],'seq':256,'lr':1e-5}))
    def new():
        model=AutoModelForSequenceClassification.from_pretrained(str(snapshot),num_labels=3,
              local_files_only=True,trust_remote_code=False,attn_implementation='eager')
        model.float().to('cuda:0')
        require(all(p.requires_grad for p in model.parameters()),'Unexpected frozen XLM parameters.')
        opt=torch.optim.AdamW(model.parameters(),lr=1e-5,weight_decay=.01)
        sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda step:max(.25,1.-.25*step))
        return model,opt,sched
    def step(model,opt,sched,batch):
        model.train();opt.zero_grad(set_to_none=True)
        inputs,y=batch;logits=model(**inputs).logits;loss=torch.nn.functional.cross_entropy(logits,y)
        require(torch.isfinite(loss).item(),'XLM-R loss is not finite.')
        loss.backward()
        g=model.roberta.embeddings.word_embeddings.weight.grad
        require(g is not None and torch.isfinite(g).all().item() and g.abs().sum().item()>0,'No valid backbone gradients.')
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        require(torch.isfinite(norm).item(),'XLM-R gradient norm is not finite.')
        opt.step();sched.step();torch.cuda.synchronize()
        return float(loss.detach())
    checkpoint=out/'xlmr_step1.PRIVATE.pt';marker=out/'xlmr_step1.ready.json'
    model,opt,sched=new()
    if marker.exists():
        require(read_json(marker)['signature']==signature,'XLM-R checkpoint input identity differs.')
        state=load_checkpoint(checkpoint);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
        restore_rng(torch,state['rng']);del state
    else:
        step(model,opt,sched,batches[0])
        save_checkpoint(checkpoint,{'model':cpu_tree(model.state_dict()),'optimizer':cpu_tree(opt.state_dict()),
             'scheduler':sched.state_dict(),'rng':rng_state(torch),'step':1,'signature':signature})
        write_json(marker,{'signature':signature})
    expected_loss=step(model,opt,sched,batches[1])
    expected_head=cpu_tree(model.classifier.out_proj.weight);expected_lr=opt.param_groups[0]['lr'];expected_step=sched.last_epoch
    # Keep only small deterministic reference probes, not a second multi-GB checkpoint.
    expected_backbone=cpu_tree(model.roberta.encoder.layer[0].attention.self.query.weight)
    del model,opt,sched;gc.collect();torch.cuda.empty_cache()
    other,opt,sched=new();state=load_checkpoint(checkpoint)
    require(state['signature']==signature and state['step']==1,'Wrong XLM-R checkpoint.')
    other.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sched.load_state_dict(state['scheduler'])
    restore_rng(torch,state['rng']);del state
    resumed_loss=step(other,opt,sched,batches[1])
    require(abs(expected_loss-resumed_loss)<=1e-5,'XLM-R resumed loss differs beyond tolerance.')
    require(torch.allclose(expected_head,other.classifier.out_proj.weight.detach().cpu(),atol=2e-6,rtol=1e-5),'XLM-R resumed head differs.')
    require(torch.allclose(expected_backbone,other.roberta.encoder.layer[0].attention.self.query.weight.detach().cpu(),atol=2e-6,rtol=1e-5),'XLM-R resumed backbone probe differs.')
    require(sched.last_epoch==expected_step and opt.param_groups[0]['lr']==expected_lr,'XLM-R optimizer/scheduler resume mismatch.')
    other.eval()
    with torch.inference_mode():logits=other(**batches[1][0]).logits
    require(logits.shape==(2,3) and torch.isfinite(logits).all().item(),'XLM-R final forward failed.')
    return {'state':'PASS','revision':asset['revision'],'actual_base_model':True,'backbone_gradients_checked':True,
        'optimizer_updates_in_uninterrupted_path':2,'checkpoint_resume_compared':True,'absolute_resumed_loss_difference':abs(expected_loss-resumed_loss),
        'max_length':256,'microbatch':2,'precision':'float32','attention':'eager','checkpoint_bytes':checkpoint.stat().st_size,
        'peak_allocated_mib':torch.cuda.max_memory_allocated(0)/2**20,'elapsed_seconds':time.monotonic()-started,
        'scope':'engineering two-step check, not tuning, complete training or production batch sizing',
        'cross_physical_gpu_resume_not_tested':True,'paper_results':False}


def local_hub_root(explicit=None):
    if explicit:return Path(explicit).expanduser().resolve()
    if os.environ.get('HF_HUB_CACHE'):return Path(os.environ['HF_HUB_CACHE']).expanduser()
    home=Path(os.environ.get('HF_HOME',str(Path.home()/'.cache'/'huggingface'))).expanduser()
    return home/'hub'


def resolve_xlmr(hub,out,allow):
    repo='FacebookAI/xlm-roberta-base';pin=Path(out)/'xlmr_source.local.json';repo_root=hub/('models--'+repo.replace('/','--'))
    if pin.exists():rev=read_json(pin)['revision']
    elif (repo_root/'refs'/'main').is_file():rev=(repo_root/'refs'/'main').read_text().strip()
    elif allow:
        from huggingface_hub import HfApi
        rev=HfApi().model_info(repo,revision='main').sha
    else:raise MissingAsset('XLM-R snapshot is absent; explicit --allow-xlmr-download is required for a download.')
    require(bool(re.fullmatch('[0-9a-f]{40}',rev or '')),'XLM-R must be pinned to a full commit hash.')
    if not pin.exists():write_json(pin,{'repo_id':repo,'revision':rev})
    path=repo_root/'snapshots'/rev
    required=['config.json','tokenizer.json','model.safetensors']
    if not all((path/f).is_file() for f in required):
        if not allow:raise MissingAsset('Pinned XLM-R snapshot is incomplete; download not authorized.')
        from huggingface_hub import snapshot_download
        # Only files needed for PyTorch+fast tokenizer; no ONNX/TF/Flax duplicate weights.
        path=Path(snapshot_download(repo,revision=rev,cache_dir=str(hub),max_workers=2,
            allow_patterns=['config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','sentencepiece.bpe.model','model.safetensors']))
    require(all((path/f).is_file() for f in required),'Downloaded XLM-R snapshot is incomplete.')
    return path


def stage_done(path,signature):
    path=Path(path);p=path/'complete.json'
    if not p.exists():return None
    d=read_json(p);require(d['signature']==signature,'Completed stage identity mismatch.')
    for name,h in d['artifacts'].items():
        require(file_sha(path/name)==h,'Completed smoke artifact changed; refusing silent reuse.')
    return read_json(path/'stage_report.json')


def mark_done(path,signature,result):
    p=Path(path);write_json(p/'stage_report.json',result)
    files={str(f.relative_to(p)):file_sha(f) for f in p.rglob('*') if f.is_file() and f.name!='complete.json' and '.partial_' not in f.name}
    write_json(p/'complete.json',{'signature':signature,'artifacts':files,'completed_at':now()})


def child_main(args):
    path=Path(args.stage_dir);result={}
    try:
        plan=read_json(path/'stage_plan.local.json');os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        if args.child=='cuda':result=cuda_probe(path)
        elif args.child in FROZEN:result=frozen_stage(Path(args.split),path,args.child,Path(plan['snapshot']),plan['asset'])
        elif args.child=='xlmr':result=xlmr_stage(Path(args.split),path,Path(plan['snapshot']),plan['asset'])
        else:raise SmokeError('Unknown child stage.')
        result.update({'device':plan['gpu'],'packages':versions()})
        write_json(path/'child_result.json',result);return 0
    except BaseException as e:
        result={'state':'INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'FAILED','error_type':type(e).__name__,
          'message':str(e) if isinstance(e,SmokeError) else 'Runtime failure; inspect PRIVATE local log. No source text is included in the shared report.'}
        write_json(path/'child_result.json',result)
        import traceback
        traceback.print_exc()
        return 130 if isinstance(e,KeyboardInterrupt) else 1


def package(out,report):
    out=Path(out);write_json(out/'model_smoke_report.json',report)
    lines=['SAUCIS REVIZYONU — MODEL/CUDA SMOKE','='*60,f"Durum: {report['state']}",
      'Final test tahmini ve makale sonucu uretilmedi. Legacy/split dosyalari degismedi.',
      f"Protocol: {report['protocol_id']}"]
    for s in report['stages']:lines.append(f"{s['stage']}: {s['state']}"+(f" ({s['message']})" if 'message' in s else ''))
    lines+=['Bu rapor sadece istenen smoke kapsamidir; tam deney izni veya tam batch kapasite testi degildir.',
            'PRIVATE dosyalar ve .pt/.joblib/.npy dosyalari paylasim ZIPine dahil DEGIL.']
    (out/'model_smoke_summary.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    z=out/'shareable_model_smoke.zip';tmp=z.with_suffix('.zip.partial')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED) as f:
        for name in ('model_smoke_report.json','model_smoke_summary.txt'):f.write(out/name,name)
    os.replace(tmp,z);print('\n'.join(lines),flush=True);print('[SHARE]',z,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--split',required=True)
    p.add_argument('--cpu-only',action='store_true')
    p.add_argument('--gpu',help='Physical nvidia-smi index or full GPU UUID; never chosen automatically.')
    p.add_argument('--allow-xlmr-download',action='store_true')
    p.add_argument('--hub-cache',default=None)
    p.add_argument('--child',choices=['cuda',*FROZEN,'xlmr'],help=argparse.SUPPRESS)
    p.add_argument('--stage-dir',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.child:return child_main(args)
    if not args.cpu_only and args.gpu is None:p.error('--gpu is required unless --cpu-only is used.')
    project=Path(args.project).expanduser().resolve();split=Path(args.split).expanduser()
    if not split.is_absolute():split=project/split
    split=split.resolve();protocol=read_json(split/'protocol.json')['protocol_id']
    root=project/'_revision_smoke'/('smoke_'+protocol+'_'+file_sha(__file__)[:10]);root.mkdir(parents=True,exist_ok=True)
    report={'schema_version':VERSION,'protocol_id':protocol,'created_at':now(),'state':'RUNNING','full_training_allowed':False,
            'stages':[],'source_files_modified':False,'final_test_predictions_created':False,
            'requested_scope':'cpu_only' if args.cpu_only else 'cpu_cuda_three_encoders_xlmr','packages':versions(),
            'limitations':['No final-test performance, no multi-seed full training and no production batch sizing.',
                           'Same-device checkpoint replay is tested; cross-physical-GPU replay is not claimed.',
                           'Smoke settings/sample sizes are engineering choices, not manuscript experiment hyperparameters.']}
    print('[OUTPUT]',root,flush=True)
    with locked(root/'.lock'):
      try:
        report['manifest_validation']=inspect_manifests(split)
        packages=versions();cpu_sig=digest(canonical({'version':file_sha(__file__),'protocol':protocol,'packages':packages,'stage':'cpu'}))
        cp=root/('cpu_'+cpu_sig[:12]);cp.mkdir(exist_ok=True)
        with locked(cp/'.lock'):
            result=stage_done(cp,cpu_sig)
            if result is None:
                print('[CPU] Small development fits, metric/cache tests and toy checkpoint...',flush=True)
                result=cpu_stage(split,cp);mark_done(cp,cpu_sig,result)
            else:print('[REUSE] Verified CPU smoke.',flush=True)
        report['stages'].append({'stage':'cpu',**result})
        if args.cpu_only:report['state']='CPU_SMOKE_PASSED';return_code=0
        else:
            gpu=safe_gpu_info(args.gpu);print('[GPU]',gpu['index'],gpu['uuid'],gpu['name'],flush=True)
            hub=local_hub_root(args.hub_cache)
            for alias in ('cuda',*FROZEN,'xlmr'):
                try:
                    if alias=='cuda':snapshot=None;asset={}
                    elif alias=='xlmr':
                        snapshot=resolve_xlmr(hub,root,args.allow_xlmr_download);asset=snapshot_manifest(snapshot,alias)
                    else:
                        repo,rev,_,_=FROZEN[alias];snapshot=hub/('models--'+repo.replace('/','--'))/'snapshots'/rev
                        asset=snapshot_manifest(snapshot,alias)
                    sig=digest(canonical({'code':file_sha(__file__),'protocol':protocol,'gpu':gpu['uuid'],
                       'alias':alias,'asset':asset,'packages':packages}))
                    stage=root/(alias+'_'+sig[:12]);stage.mkdir(exist_ok=True)
                    result=stage_done(stage,sig)
                    if result is not None:print('[REUSE]',alias,'verified.',flush=True)
                    else:
                        write_json(stage/'stage_plan.local.json',{'gpu':gpu,'snapshot':str(snapshot) if snapshot else None,'asset':asset})
                        env=dict(os.environ);env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
                        env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['TOKENIZERS_PARALLELISM']='false'
                        env['OMP_NUM_THREADS']='2';env['MKL_NUM_THREADS']='2';env['OPENBLAS_NUM_THREADS']='2';env['PYTHONHASHSEED']='42'
                        command=[sys.executable,str(Path(__file__).resolve()),'--split',str(split),'--child',alias,'--stage-dir',str(stage)]
                        print('[SMOKE]',alias,'(one child process, selected GPU only)',flush=True)
                        log=stage/'runtime.PRIVATE.log'
                        with log.open('w',encoding='utf-8') as fh:
                            proc=subprocess.Popen(command,env=env,stdout=fh,stderr=subprocess.STDOUT,start_new_session=True)
                            try:code=proc.wait()
                            except KeyboardInterrupt:
                                # Give the child a chance to finish its current atomic write and report.
                                os.killpg(proc.pid,signal.SIGINT)
                                try:proc.wait(timeout=120)
                                except subprocess.TimeoutExpired:
                                    os.killpg(proc.pid,signal.SIGTERM);proc.wait(timeout=30)
                                raise
                        require((stage/'child_result.json').exists(),f'{alias}: child stopped before writing a report; inspect PRIVATE log.')
                        result=read_json(stage/'child_result.json')
                        if code!=0 or result['state']!='PASS':
                            report['stages'].append({'stage':alias,**result});print('[LOCAL LOG]',log,flush=True)
                            raise SmokeError(f'{alias} smoke did not pass. Do not launch full experiments.')
                        result['asset']=asset;mark_done(stage,sig,result)
                    report['stages'].append({'stage':alias,**result})
                    print('[PASS]',alias,flush=True)
                except MissingAsset as e:
                    report['stages'].append({'stage':alias,'state':'NOT_RUN','message':str(e)})
            allpass=all(s['state']=='PASS' for s in report['stages']) and len(report['stages'])==6
            report['state']='MODEL_SMOKE_PASSED' if allpass else 'MODEL_SMOKE_INCOMPLETE';return_code=0 if allpass else 2
      except BaseException as e:
        report['state']='INTERRUPTED' if isinstance(e,KeyboardInterrupt) else 'MODEL_SMOKE_BLOCKED'
        report['error_type']=type(e).__name__;report['message']=str(e) if isinstance(e,SmokeError) else 'Failure during smoke. Existing source/split files were not changed.'
        return_code=130 if isinstance(e,KeyboardInterrupt) else 1
        if not isinstance(e,(KeyboardInterrupt,SmokeError)):
            import traceback
            (root/'error.PRIVATE.log').write_text(traceback.format_exc(),encoding='utf-8')
            print('[LOCAL LOG]',root/'error.PRIVATE.log',flush=True)
      finally:package(root,report)
    return return_code

if __name__=='__main__':sys.exit(main())
