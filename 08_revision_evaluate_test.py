#!/usr/bin/env python3
"""SAUCIS step08: freeze validation choices, then evaluate immutable TEST manifests.

No fitting, optimization, calibration, threshold selection, or model selection on
TEST. Source files are read-only. All 1200 linear choices and 45 XLM-R choices
are locked before the first TEST example/label is loaded. Failed candidates are
never skipped to improve scores. Predictions are committed in restartable shards;
KZ extreme-rating views reuse the corresponding binary predictions exactly.

This stage reports descriptive metrics and sample SD over five splits. Paired
bootstrap/significance/error interpretation are a separate analysis of these
predictions, not a reason to retrain or reselect models.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager, ExitStack
from dataclasses import dataclass
import csv, gc, hashlib, importlib.util, io, json, math, os
from pathlib import Path
import signal, sqlite3, subprocess, sys, time, traceback, zipfile

VERSION = '1.0.0'
TRAINER = ('07_revision_train_xlmr.py','9e6d39a4a02494f95b09211b13275ad87cf0954b32547ad3b1ae5bde67b2b9e4')
SEEDS = (42,43,44,45,46)
REPS = ('word','char','minilm','labse','bge_m3')
SHARD = 1024
SETTINGS = dict(primary_metric='macro_f1', class_zero_division=0, labels='fixed_task_labels',
    predictions='native_classifier_predict_or_argmax_XLM_logits', calibration=False,
    model_selection_on_test=False, refit=False, xlmr_precision='fp32', xlmr_microbatch=4,
    xlmr_max_length=256, xlmr_padding='fixed_256', xlmr_attention='eager', TF32=False,
    prediction_shard_rows=SHARD, extreme='reuse_binary_KZ_test_predictions',
    summary='arithmetic_mean_and_sample_SD_over_five_splits_not_pooled_independent_rows',
    statistical_tests_in_this_stage=False)
_STOP = False
class EvaluationError(RuntimeError): pass
class Stopped(EvaluationError): pass
def need(ok,msg):
    if not ok: raise EvaluationError(msg)
def now():
    from datetime import datetime,timezone
    return datetime.now(timezone.utc).isoformat()
def canonical(x): return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x): return hashlib.sha256(x.encode()).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def safe(root,rel):
    r=Path(rel);need(not r.is_absolute() and '..' not in r.parts,'Unsafe relative path.')
    p=Path(root)/r;need(p.resolve().is_relative_to(Path(root).resolve()),'Path escapes artifact directory.');return p
def atomic(p,writer,binary=True):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_name(p.name+f'.partial_{os.getpid()}_{time.time_ns()}')
    with t.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf-8'})) as f:
        writer(f);f.flush();os.fsync(f.fileno())
    os.replace(t,p)
    fd=os.open(p.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
def write(p,x):atomic(p,lambda f:f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'),False)
def write_npz(p,a):
    import numpy as np
    atomic(p,lambda f:np.savez_compressed(f,**a))
def csvwrite(p,rows,cols):
    b=io.StringIO(newline='');w=csv.DictWriter(b,fieldnames=cols,extrasaction='ignore');w.writeheader();w.writerows(rows)
    atomic(p,lambda f:f.write(b.getvalue()),False)
def desc(p):return dict(bytes=Path(p).stat().st_size,sha256=sha(p))
class Verifier:
    def __init__(self):self.cache={}
    def check(self,p,d):
        p=Path(p);need(p.is_file(),'Committed artifact missing; do not delete or regenerate silently.')
        s=p.stat();need(s.st_size==d['bytes'],'Artifact size differs.')
        k=(str(p.resolve()),s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
        if k not in self.cache:self.cache[k]=sha(p)
        need(self.cache[k]==d['sha256'],'Artifact checksum differs; evaluation blocked.')
def commit(folder,sig,names):write(Path(folder)/'complete.json',dict(signature=sig,files={n:desc(safe(folder,n)) for n in names}))
def ready(folder,sig,v=None):
    p=Path(folder)/'complete.json'
    if not p.exists():return False
    d=read(p);need(d['signature']==sig and d['files'],'Completed output belongs to a different input/selection.')
    v=v or Verifier()
    for n,h in d['files'].items():v.check(safe(folder,n),h)
    return True
@contextmanager
def lock(path,shared=False):
    import fcntl
    p=Path(path)
    if not shared:p.parent.mkdir(parents=True,exist_ok=True)
    need(not shared or p.is_file(),'Expected source-run lock file missing.')
    with p.open('r' if shared else 'a') as f:
        try:fcntl.flock(f, (fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
        except BlockingIOError as e:raise EvaluationError('Another process is modifying this source/output run.') from e
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)
def stop(signum,frame):
    global _STOP;_STOP=True
def check_stop(path=None):
    if _STOP or (path and Path(path).exists()):raise Stopped('Stopped; committed prediction shards are retained.')
def load_trainer(project):
    p=Path(project)/TRAINER[0];need(p.is_file() and sha(p)==TRAINER[1],'Reviewed step07 helper missing/changed.')
    name='_saucis_evaluation_trainer'
    if name not in sys.modules:
        s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m)
    return sys.modules[name]
def metadata(L,split,seed,sid):
    with L.ro_db(Path(split)/f'seed_{seed}'/'memberships.local.sqlite3') as db:
        row=db.execute('SELECT role,task,n_rows,metadata_json FROM sets WHERE set_id=?',(sid,)).fetchone()
    need(row is not None,'Unknown set in approved recipe.')
    return dict(set_id=sid,role=row[0],task=row[1],rows=row[2],metadata=json.loads(row[3]))
def mkey(seed,ctx,rep,clf):return f'seed_{seed}/'+digest(ctx)[:16]+'/'+rep+'_'+clf

def audit_xlmr_reports(X,root,base,contexts,v,production=True):
    root=Path(root);r=read(root/'xlmr_training_report.json');spec=read(root/'root_spec.json');plan=read(root/'training_plan.json')
    expected_jobs=len(plan['jobs']);expected_choices=expected_jobs//2
    need((not production or expected_jobs==90) and r['state']=='XLMR_MODELS_READY' and r['completed_jobs']==expected_jobs and r['selected_models']==expected_choices and r['completed_epoch_evaluations']==3*expected_jobs,'XLM full scope incomplete.')
    need(not r['pending_jobs'] and not r['final_test_predictions_created'] and not r['refit_train_plus_validation'],'Unexpected XLM test use/refit/pending job.')
    need(spec==plan['root_spec'] and digest(canonical(spec))==r['root_spec_sha256'],'XLM root identity mismatch.')
    need(spec['runner_sha256']==TRAINER[1] and spec['settings']==r['settings']==X.SETTINGS,'XLM scientific settings/producer differ.')
    for k in ('protocol_id','corpus_sha256','split_report_sha256','feature_report_sha256','smoke_report_sha256'):
        need(spec[k]==base[k],'XLM/source identity mismatch.')
    jobs={j['key']:j for j in plan['jobs']};need(len(jobs)==expected_jobs,'Duplicate XLM plan keys.')
    grouped=defaultdict(list)
    for c in r['candidates']:
        job=jobs[c['key']];need(c['signature']==job['signature'],'XLM candidate signature mismatch.')
        folder=safe(root/'jobs',c['key']);done=read(folder/'complete.json')
        need(done['signature']==job['signature'],'Candidate completion signature mismatch.')
        v.check(folder/'candidate_report.json',done['files']['candidate_report.json'])
        need(read(folder/'candidate_report.json')==c,'Aggregate/candidate report differs.')
        need(c['settings']==X.SETTINGS and not c['final_test_read'] and not c['refit_train_plus_validation'],'Unsafe candidate.')
        need(c['optimizer_steps']==job['optimizer_steps'],'Incomplete candidate steps.')
        for e in c['epochs']:
            vals=e['validation']['by_language']
            need(set(vals)==set(contexts[c['seed']][c['context_id']]['validation_languages']),'Unauthorized validation language.')
            need(sum(x['macro_f1'] for x in vals.values())/len(vals)==e['validation']['selection_score'],'Wrong equal-language/source selection score.')
            ef=folder/f'epoch_{e["epoch"]}';ec=read(ef/'complete.json');v.check(ef/'epoch_report.json',ec['files']['epoch_report.json'])
            need(read(ef/'epoch_report.json')==e and not e['final_test_read'],'Epoch report mismatch.')
        best=X.candidate_best(c['epochs']);need(c['best_epoch']==best['epoch'],'Invalid epoch choice.')
        grouped[c['seed'],c['context_id']].append(c)
    need(len(grouped)==expected_choices and len(r['selections'])==expected_choices,'XLM selection count differs.')
    models=[]
    for selected in r['selections']:
        seed,ctx=selected['seed'],selected['context_id'];pair=grouped[seed,ctx]
        need(pair[0]['initial_model_sha256']==pair[1]['initial_model_sha256'],'Unequal LR initializations.')
        need(X.select_context(pair)==selected,'XLM choice violates validation/tie rule.')
        sp=root/'selections'/f'seed_{seed}'/(X.re_safe(ctx)+'.json');need(read(sp)==selected,'Selected file differs.')
        job=jobs[selected['selected_job_key']];c=next(c for c in pair if c['key']==job['key'])
        e=X.check_epoch(root/'jobs'/job['key']/f'epoch_{selected["epoch"]}',job,selected['epoch'],c['data_signature'],v)
        need(e and e['weights']['sha256']==selected['selected_model_sha256'],'Selected weights mismatch.')
        models.append(dict(model_key=mkey(seed,ctx,'xlmr','finetuned'),kind='xlmr',seed=seed,context_id=ctx,
            representation='xlmr',classifier='finetuned',task=selected['task'],context=contexts[seed][ctx],
            model_file=selected['selected_model_relative_path'],model_descriptor=e['weights'],selection=selected,
            validation_folder=f'jobs/{job["key"]}/epoch_{selected["epoch"]}/validation',
            validation_report_sha256=sha(root/'jobs'/job['key']/f'epoch_{selected["epoch"]}'/'validation/validation_report.json')))
    need(len({m['model_key'] for m in models})==expected_choices,'Duplicate XLM selected model.')
    return models,r,spec

def audit_linear_models(L,P,root,base,contexts,fr,split,v):
    root=Path(root);P.audit_linear_directory(root,base)
    spec=read(root/'root_spec.json');models=[]
    for seed,ctxs in sorted(contexts.items()):
        for ctx,c in ctxs.items():
            tm=metadata(L,split,seed,c['train_set']);vm=metadata(L,split,seed,c['validation_set'])
            brief=lambda d:dict(set_id=d['set_id'],role=d['role'],task=d['task'],rows=d['rows'],manifest_sha256=d['metadata']['manifest_sha256'])
            for rep in REPS:
                folder=root/f'seed_{seed}'/L.safe_name(ctx)/rep
                rs=read(folder/'run_spec.json')
                expected=dict(root_spec_sha256=digest(canonical(spec)),seed=seed,context=c,train=brief(tm),validation=brief(vm),representation=rep,
                    feature_spec_sha256=fr['banks'][rep]['spec_sha256'] if rep in L.DIMENSIONS else None,C_grid=list(L.C_GRID))
                need(rs==expected,'Linear representation identity differs from manifests.')
                need(L.verified_bundle(folder,rs),'Incomplete linear representation.')
                transformer=None
                if rep in ('word','char'):
                    tid={'run_spec':rs,'stage':'train_only_tfidf'}
                    need(L.verified_bundle(folder/'transformer',tid),'Missing train-only transformer.')
                    tf=folder/'transformer/vectorizer.PRIVATE.joblib'
                    ti=read(folder/'transformer/transformer_report.json')
                    need(ti['validation_used_for_vocabulary_or_idf'] is False and ti['final_test_accessed'] is False,'Unsafe transformer.')
                    transformer=dict(file=str(tf.relative_to(root)),descriptor=desc(tf),signature=ti['signature'],parameters=ti['parameters'])
                for clf in ('lr','svm'):
                    trials=[]
                    for i,C in enumerate(L.C_GRID):
                        cf=folder/clf/f'C_{i}';ident={'run_spec':rs,'classifier':clf,'C':float(C)};stamp=read(cf/'complete.json')
                        need(stamp['identity_sha256']==digest(canonical(ident)),'C candidate identity mismatch.')
                        v.check(cf/'candidate_report.json',stamp['outputs']['candidate_report.json'])
                        a=read(cf/'candidate_report.json');need(a['state']=='CANDIDATE_READY' and a['converged'] and not a['final_test_predictions_created'],'Invalid candidate in selection.')
                        need(a['C']==C and a['classifier']==clf and a['seed']==seed and a['context_id']==ctx,'Mixed C candidates.')
                        vals=a['validation']['by_language'];need(set(vals)==set(c['validation_languages']),'Unauthorized validation labels.')
                        need(sum(z['macro_f1'] for z in vals.values())/len(vals)==a['validation']['selection_score'],'Wrong linear selection metric.')
                        trials.append(a)
                    b=L.winner(trials);s=read(folder/clf/'selected_config.json');ix=list(L.C_GRID).index(b['C']);cf=folder/clf/f'C_{ix}'
                    need(s['selected_C']==b['C'] and s['selection_score']==b['validation']['selection_score'] and s['validation']==b['validation'],'Saved linear choice differs.')
                    need(not s['final_test_used_for_selection'] and not s['refit_train_plus_validation'],'Unsafe linear selection.')
                    need(L.accepted_report(cf,{'run_spec':rs,'classifier':clf,'C':b['C']},clf,b['C'],seed,c) is not None,'Missing selected candidate/provenance.')
                    mf=cf/'classifier.PRIVATE.joblib';pf=cf/'validation_predictions.PRIVATE.npz'
                    need(sha(mf)==s['selected_model_sha256'],'Selected linear weight hash differs.')
                    models.append(dict(model_key=mkey(seed,ctx,rep,clf),kind='linear',seed=seed,context_id=ctx,representation=rep,classifier=clf,
                        task=s['task'],context=c,model_file=str(mf.relative_to(root)),model_descriptor=desc(mf),selection=s,
                        transformer=transformer,candidate_parameters=b['parameters'],feature_spec_sha256=rs['feature_spec_sha256'],
                        validation_file=str(pf.relative_to(root)),validation_descriptor=desc(pf)))
    expected_count=sum(len(c) for c in contexts.values())*len(REPS)*2
    need(len(models)==expected_count and len({m['model_key'] for m in models})==expected_count,'Linear model coverage mismatch.')
    return models

def build_plan(models,recipes,meta,production=True):
    out=[]
    for model in models:
        es=[e for e in recipes[model['seed']] if e['training_context_id']==model['context_id']]
        need(es,'Model has no evaluation recipe.')
        jobs=[]
        for e in es:
            need(e['test_labels_must_not_select_models'] is True,'Unsafe recipe.')
            for x,y in [('train_set','train_set'),('validation_set','validation_set')]:need(e[x]==model['context'][y],'Recipe development mismatch.')
            d=meta(model['seed'],e['test_set']);need(d['role']=='test' and d['task']==model['task'],'Wrong test role/task.')
            jobs.append(dict(recipe=e,test_manifest=d,evaluation_key=model['model_key']+'/'+digest(e['experiment_id'])[:16]))
        need(len({j['evaluation_key'] for j in jobs})==len(jobs),'Repeated evaluation recipe.')
        out.append(dict(model=model,evaluations=jobs))
    counts=Counter(j['model']['kind'] for j in out for _ in j['evaluations'])
    if production:need(counts=={'linear':2700,'xlmr':120},'Evaluation matrix differs from approved 2820 cells.')
    return out

@dataclass
class TestPart:
    row_ids:object;sample_ids:object;group_ids:object;texts:list;targets:object
    languages:object;domains:object;text_hashes:object;ratings:object;metadata:dict;task:str;role:str='test'
    def __len__(self):return len(self.row_ids)

def load_test(L,split,seed,sid,frozen):
    need(frozen.get('state')=='SELECTIONS_LOCKED_BEFORE_TEST' and frozen.get('model_count',0)>0,'Cannot read TEST before all selections are locked.')
    import numpy as np
    with L.ro_db(Path(split)/f'seed_{seed}'/'memberships.local.sqlite3') as db:
        m=db.execute('SELECT role,task,n_rows,metadata_json FROM sets WHERE set_id=?',(sid,)).fetchone()
        need(m and m[0]=='test','Test reader refuses non-test roles.')
        db.execute('ATTACH DATABASE ? AS corpus',((Path(split)/'corpus.local.sqlite3').resolve().as_uri()+'?mode=ro',))
        rows=db.execute('''SELECT r.row_id,r.sample_id,r.group_id,r.model_text,m.target,r.language,r.domain,r.label,r.rating
            FROM members m JOIN corpus.records r USING(row_id) WHERE m.set_id=? ORDER BY r.row_id''',(sid,)).fetchall()
    role,task,n,raw=m;info=json.loads(raw);need(len(rows)==n and n>0,'Test count mismatch.')
    need(len({r[0] for r in rows})==n and len({r[1] for r in rows})==n,'Duplicate test identities.')
    h=hashlib.sha256()
    for r in rows:
        h.update((r[1]+':'+str(r[4])+'\n').encode())
        expected=r[7] if task=='three_class' else (0 if r[7]==0 else 1)
        need((task=='three_class' or r[7] in (0,2)) and expected==r[4],'Test label schema mismatch.')
    need(h.hexdigest()==info['manifest_sha256'],'Test membership hash differs.')
    need(dict(Counter(r[5] for r in rows))==info['language_counts'],'Test language counts differ.')
    if sid.startswith('extreme:'):need(all(r[5]=='kz' and r[8] in (1,5) for r in rows),'Extreme view includes unauthorized ratings.')
    return TestPart(np.array([r[0] for r in rows],np.int64),np.array([r[1] for r in rows],dtype='S64'),np.array([r[2] for r in rows],dtype='S64'),
        [r[3] for r in rows],np.array([r[4] for r in rows],np.int64),np.array([r[5] for r in rows]),np.array([r[6] for r in rows]),
        np.array([digest(r[3]) for r in rows],dtype='S64'),np.array([r[8] if r[8] is not None else -1 for r in rows],np.int16),info,task)

def check_disjoint(part,tr,va):
    need(set(part.group_ids).isdisjoint(set(tr.group_ids.astype('S64'))) and set(part.group_ids).isdisjoint(set(va.group_ids.astype('S64'))),'TEST text groups overlap train/validation.')
    need(set(part.row_ids).isdisjoint(tr.row_ids) and set(part.row_ids).isdisjoint(va.row_ids),'TEST record overlaps development.')

def checked_output(output,n,k,kind):
    import numpy as np
    need('y_pred' in output and output['y_pred'].shape==(n,),'Prediction row shape differs.')
    need(set(output['y_pred'])<=set(range(k)),'Predicted class outside task.')
    scorekey='logits' if kind=='xlmr' else 'decision_scores'
    need(scorekey in output,'Missing native scores.')
    scores=output[scorekey];expected=(n,k) if kind=='xlmr' or k>2 else (n,)
    need(scores.shape==expected and np.isfinite(scores).all(),'Invalid score shape/values.')
    if 'probabilities' in output:
        p=output['probabilities'];need(p.shape==(n,k) and np.isfinite(p).all() and np.all(p>=0) and np.all(p<=1) and np.allclose(p.sum(1),1,atol=1e-6),'Invalid probabilities.')
    if kind=='xlmr':need(np.array_equal(output['y_pred'],np.argmax(scores,axis=1)),'XLM prediction is not logits argmax.')

def metric_pack(y, pred, labels):
    """Fixed-class metrics from one confusion matrix; cross-tested against sklearn.

    Explicit zero_division=0 for an absent class. No fitting/threshold adjustment.
    This avoids recomputing the same confusion statistics for each reported metric.
    """
    import numpy as np
    y=np.asarray(y);pred=np.asarray(pred);k=len(labels)
    need(labels==list(range(k)) and k in (2,3),'Unexpected metric class order.')
    need(y.ndim==pred.ndim==1 and len(y)>0 and len(y)==len(pred),'Invalid metric arrays.')
    need(np.isin(y,labels).all() and np.isin(pred,labels).all(),'Metric class outside task.')
    cm=np.bincount((y.astype(np.int64)*k+pred.astype(np.int64)),minlength=k*k).reshape(k,k)
    tp=np.diag(cm).astype(float);support=cm.sum(1);predicted=cm.sum(0)
    divide=lambda a,b:np.divide(a,b,out=np.zeros(k,dtype=float),where=b!=0)
    precision=divide(tp,predicted);recall=divide(tp,support);f1=divide(2*tp,support+predicted)
    names=['negative','neutral','positive'] if k==3 else ['negative','positive']
    report={name:{'precision':float(precision[i]),'recall':float(recall[i]),'f1-score':float(f1[i]),'support':float(support[i])} for i,name in enumerate(names)}
    accuracy=float(tp.sum()/len(y))
    report['accuracy']=accuracy
    report['macro avg']=dict(precision=float(precision.mean()),recall=float(recall.mean()),**{'f1-score':float(f1.mean()),'support':float(len(y))})
    report['weighted avg']=dict(precision=float(np.dot(precision,support)/len(y)),recall=float(np.dot(recall,support)/len(y)),**{'f1-score':float(np.dot(f1,support)/len(y)),'support':float(len(y))})
    return dict(n=len(y),accuracy=accuracy,macro_f1=float(f1.mean()),weighted_f1=float(np.dot(f1,support)/len(y)),macro_precision=float(precision.mean()),macro_recall=float(recall.mean()),class_report=report,class_order=names,confusion_matrix=cm.tolist())

def prediction_spec(locksha,model,evaluation):
    return dict(selection_lock_sha256=locksha,model_key=model['model_key'],model_sha256=model['model_descriptor']['sha256'],
        evaluation_key=evaluation['evaluation_key'],recipe=evaluation['recipe'],test_manifest=evaluation['test_manifest'],settings=SETTINGS)

def metadata_payload(part,lo,hi):
    import numpy as np
    texts=part.texts[lo:hi]
    return dict(row_ids=part.row_ids[lo:hi],sample_ids=part.sample_ids[lo:hi],group_ids=part.group_ids[lo:hi],
        y_true=part.targets[lo:hi],languages=part.languages[lo:hi],domains=part.domains[lo:hi],ratings=part.ratings[lo:hi],
        character_lengths=np.array([len(t) for t in texts],np.int32),word_lengths=np.array([len(t.split()) for t in texts],np.int32))

def evaluate_set(L,folder,spec,model,part,predict,execution,stop_file=None,after_shard=None):
    """predict(lo,hi) receives features only; labels never enter the prediction callable."""
    import numpy as np
    folder=Path(folder);sig=digest(canonical(spec));v=Verifier()
    need(model['task']==part.task and part.role=='test','Wrong task/role at evaluation boundary.')
    if ready(folder,sig,v):
        r=read(folder/'evaluation_report.json');need(r['signature']==sig,'Evaluation report mismatch.');return r,False
    folder.mkdir(parents=True,exist_ok=True)
    if (folder/'evaluation_spec.json').exists():need(read(folder/'evaluation_spec.json')==spec,'Partial evaluation identity differs.')
    else:write(folder/'evaluation_spec.json',spec)
    names=['evaluation_spec.json','evaluation_report.json'];preds=[];aux=defaultdict(list);executions=[]
    labs=L.task_labels(part.task)
    for i,lo in enumerate(range(0,len(part),SHARD)):
        check_stop(stop_file);hi=min(len(part),lo+SHARD);sf=folder/f'shard_{i:05d}'
        ss=digest(canonical(dict(signature=sig,index=i,lo=lo,hi=hi)))
        if ready(sf,ss,v):
            sm=read(sf/'shard_report.json')
            with np.load(sf/'predictions.PRIVATE.npz',allow_pickle=False) as f:a={k:f[k] for k in f.files}
        else:
            begun=time.monotonic();output=predict(lo,hi);checked_output(output,hi-lo,len(labs),model['kind'])
            need(not (set(output)&set(metadata_payload(part,lo,hi))),'Predictor returned labels/identity instead of features.')
            a=dict(metadata_payload(part,lo,hi),**output)
            sf.mkdir(parents=True,exist_ok=True);write_npz(sf/'predictions.PRIVATE.npz',a)
            sm=dict(signature=ss,lo=lo,hi=hi,rows=hi-lo,prediction_seconds=time.monotonic()-begun,execution=execution)
            write(sf/'shard_report.json',sm);commit(sf,ss,['predictions.PRIVATE.npz','shard_report.json'])
            if after_shard:after_shard(i)
        need(sm['signature']==ss and sm['lo']==lo and sm['hi']==hi,'Shard range identity differs.')
        for k,z in metadata_payload(part,lo,hi).items():need(np.array_equal(a[k],z),'Predictions/TEST identity or label alignment differs.')
        checked_output(a,hi-lo,len(labs),model['kind']);preds.append(a['y_pred']);executions.append(sm)
        for k in ('zero_vector','vocabulary_token_count','analyzed_token_count','token_lengths'):
            if k in a:aux[k].append(a[k])
        names.extend([f'shard_{i:05d}/complete.json',f'shard_{i:05d}/predictions.PRIVATE.npz',f'shard_{i:05d}/shard_report.json'])
        if i%20==0 or hi==len(part):print(f'[TEST] {model["model_key"]} {spec["recipe"]["experiment_id"]}: {hi:,}/{len(part):,}',flush=True)
    prediction=np.concatenate(preds)
    metrics=metric_pack(part.targets,prediction,labs)
    strata={}
    for axis,values in [('domain',part.domains),('language',part.languages),('rating',part.ratings)]:
        if axis=='rating' and np.all(values==-1):continue
        strata[axis]={str(x):metric_pack(part.targets[values==x],prediction[values==x],labs) for x in sorted(set(values))}
    diagnostic={}
    if 'zero_vector' in aux:diagnostic['zero_vector_count']=int(np.concatenate(aux['zero_vector']).sum())
    if 'analyzed_token_count' in aux:
        total=int(np.concatenate(aux['analyzed_token_count']).sum());matched=int(np.concatenate(aux['vocabulary_token_count']).sum())
        diagnostic.update(analyzed_ngram_occurrences=total,in_vocabulary_occurrences=matched,occurrence_coverage=matched/total if total else None)
    r=dict(state='TEST_EVALUATION_READY',signature=sig,evaluation_key=spec['evaluation_key'],model_key=model['model_key'],seed=model['seed'],
        context_id=model['context_id'],representation=model['representation'],classifier=model['classifier'],task=part.task,
        experiment_id=spec['recipe']['experiment_id'],test_set=spec['recipe']['test_set'],test_manifest_sha256=part.metadata['manifest_sha256'],
        model_sha256=model['model_descriptor']['sha256'],selection_score_validation=model['selection']['selection_score'],
        selected_C=model['selection'].get('selected_C'),selected_lr=model['selection'].get('learning_rate'),selected_epoch=model['selection'].get('epoch'),
        metrics=metrics,strata=strata,diagnostics=diagnostic,shards=len(preds),prediction_seconds=sum(s['prediction_seconds'] for s in executions),
        device_executions=list({canonical(s['execution']):s['execution'] for s in executions}.values()),
        trained_or_selected_on_test=False,model_file_modified=False,created_at=now())
    write(folder/'evaluation_report.json',r);commit(folder,sig,names);return r,True

def read_predictions(folder,expected_sig):
    import numpy as np
    need(ready(folder,expected_sig),'Parent prediction result missing/corrupt.')
    r=read(Path(folder)/'evaluation_report.json');out=defaultdict(list)
    for i in range(r['shards']):
        with np.load(Path(folder)/f'shard_{i:05d}/predictions.PRIVATE.npz',allow_pickle=False) as z:
            for k in z.files:out[k].append(z[k])
    return {k:np.concatenate(v) for k,v in out.items()}

def extreme_predictor(parent,part):
    import numpy as np
    ids=parent['row_ids'];need(np.all(ids[1:]>ids[:-1]),'Parent rows not strictly ordered.')
    ix=np.searchsorted(ids,part.row_ids);need(np.all(ix<len(ids)),'Extreme IDs absent from binary result.')
    for k,v in [('row_ids',part.row_ids),('sample_ids',part.sample_ids),('group_ids',part.group_ids),('y_true',part.targets)]:need(np.array_equal(parent[k][ix],v),'Extreme membership/label differs from binary parent.')
    keys=('y_pred','decision_scores','probabilities','logits','zero_vector','vocabulary_token_count','analyzed_token_count','token_lengths')
    return lambda lo,hi:{k:parent[k][ix[lo:hi]] for k in keys if k in parent}

class LinearFeatures:
    """One frozen bank or one lexical source-context cache; no fit methods."""
    def __init__(self,L,features,fr,base):self.L=L;self.features=Path(features);self.fr=fr;self.base=base;self.bank=None;self.rep=None;self.lexkey=None;self.cache={}
    def dense(self,rep,part):
        if self.rep!=rep:
            self.bank=None;gc.collect();self.rep=rep;e=self.fr['banks'][rep]
            self.bank=self.L.FrozenBank(safe(self.features,e['bank_directory']),e,self.base['corpus_sha256'],self.base['protocol_id'])
        return self.bank.development(part)
    def lexical(self,key,vector,part):
        import numpy as np
        if self.lexkey!=key:self.cache.clear();self.lexkey=key
        ps=part.metadata['manifest_sha256']
        if ps not in self.cache:
            X=vector.transform(part.texts);need(np.isfinite(X.data).all(),'Nonfinite TF-IDF TEST features.')
            analyzer=vector.build_analyzer();vocab=vector.vocabulary_;total=[];known=[]
            for text in part.texts:
                a=analyzer(text);total.append(len(a));known.append(sum(t in vocab for t in a))
            self.cache[ps]=(X,dict(zero_vector=(np.diff(X.indptr)==0),analyzed_token_count=np.asarray(total,np.int32),vocabulary_token_count=np.asarray(known,np.int32)))
        return self.cache[ps]

def validate_linear_reload(L,model,clf,vector,bank,tr,va,root):
    import numpy as np
    v=Verifier();v.check(safe(root,model['validation_file']),model['validation_descriptor'])
    with np.load(safe(root,model['validation_file']),allow_pickle=False) as f:
        need(np.array_equal(f['row_ids'],va.row_ids) and np.array_equal(f['sample_ids'],va.sample_ids.astype('S64')) and np.array_equal(f['y_true'],va.targets),'Stored validation alignment differs.')
        pred=f['y_pred'][:64]
    X=vector.transform(va.texts[:64]) if vector is not None else bank.dense(model['representation'],va)[:64]
    need(np.array_equal(clf.predict(X),pred),'Reloaded linear model failed its saved validation sentinel.')
    need(clf.classes_.tolist()==L.task_labels(va.task),'Classifier class order differs.')
    need(L.serial_params(clf)==model['candidate_parameters'],'Reloaded classifier parameters differ.')


def run_linear(X,L,plan,paths,out,freeze,locksha,fr,base,stop_file=None,after_shard=None):
    import numpy as np,joblib
    from threadpoolctl import threadpool_limits
    cache=LinearFeatures(L,paths['features'],fr,base);parts={};devkey=None;tr=va=None
    ordered=sorted((j for j in plan if j['model']['kind']=='linear'),key=lambda j:(REPS.index(j['model']['representation']),j['model']['seed'],j['model']['context_id'],j['model']['classifier']))
    with threadpool_limits(limits=4):
      for j in ordered:
        check_stop(stop_file);m=j['model'];mf=safe(paths['linear'],m['model_file']);Verifier().check(mf,m['model_descriptor'])
        if all(ready(safe(out/'evaluations',e['evaluation_key']),digest(canonical(prediction_spec(locksha,m,e)))) for e in j['evaluations']):
            print('[REUSE]',m['model_key'],'all TEST results verified',flush=True);continue
        key=(m['seed'],m['context_id'])
        if devkey!=key:
            devkey=key;parts.clear();tr=L.load_development(paths['split'],m['seed'],m['context']['train_set']);va=L.load_development(paths['split'],m['seed'],m['context']['validation_set']);L.validate_pair(tr,va,m['context'])
        clf=joblib.load(mf);vector=None
        if m['transformer']:
            t=m['transformer'];tf=safe(paths['linear'],t['file']);Verifier().check(tf,t['descriptor']);vector=joblib.load(tf)
            need(L.vector_signature(vector)==t['signature'],'TF-IDF vocabulary/IDF changed.')
        validate_linear_reload(L,m,clf,vector,cache,tr,va,paths['linear'])
        efmap={e['recipe']['experiment_id']:e for e in j['evaluations']}
        for e in sorted(j['evaluations'],key=lambda e:bool(e['recipe']['reuse_fitted_model_from'])):
            check_stop(stop_file);spec=prediction_spec(locksha,m,e);folder=safe(out/'evaluations',e['evaluation_key'])
            if ready(folder,digest(canonical(spec))):continue
            sid=e['recipe']['test_set']
            if sid not in parts:parts[sid]=load_test(L,paths['split'],m['seed'],sid,freeze)
            part=parts[sid];check_disjoint(part,tr,va)
            reuse=e['recipe']['reuse_fitted_model_from']
            if reuse:
                parent=efmap[reuse];payload=read_predictions(safe(out/'evaluations',parent['evaluation_key']),digest(canonical(prediction_spec(locksha,m,parent))))
                predict=extreme_predictor(payload,part);execution=dict(device='CPU',reused_prediction_recipe=reuse)
            else:
                if vector is not None:feat,aux=cache.lexical((m['seed'],m['context_id'],m['representation'],m['transformer']['signature']),vector,part)
                else:
                    feat=cache.dense(m['representation'],part);aux={'token_lengths':cache.bank.lengths[part.row_ids-1]}
                def predict(lo,hi):
                    z=feat[lo:hi];p={'y_pred':clf.predict(z).astype(np.int64),'decision_scores':np.asarray(clf.decision_function(z))}
                    if m['classifier']=='lr':p['probabilities']=clf.predict_proba(z)
                    return p|{k:a[lo:hi] for k,a in aux.items()}
                execution=dict(device='CPU',threads=4,model_sha256=m['model_descriptor']['sha256'])
            evaluate_set(L,folder,spec,m,part,predict,execution,stop_file,after_shard)
        if vector is not None:need(L.vector_signature(vector)==m['transformer']['signature'],'Inference mutated TF-IDF.')
        need(sha(mf)==m['model_descriptor']['sha256'],'Source model changed during evaluation.')
        del clf,vector;gc.collect()


def logit_predict(model,input_ids,mask,device,microbatch=4):
    import numpy as np,torch
    values=[];model.eval()
    with torch.inference_mode():
        for lo in range(0,len(input_ids),microbatch):
            check_stop();hi=min(len(input_ids),lo+microbatch)
            x=torch.as_tensor(np.asarray(input_ids[lo:hi]),device=device,dtype=torch.long)
            a=torch.as_tensor(np.asarray(mask[lo:hi]),device=device,dtype=torch.long)
            logits=model(input_ids=x,attention_mask=a).logits
            need(torch.isfinite(logits).all().item(),'Nonfinite XLM inference logits.')
            values.append(logits.float().cpu().numpy())
    z=np.concatenate(values);p=np.exp(z-z.max(axis=1,keepdims=True));p/=p.sum(axis=1,keepdims=True)
    return dict(logits=z,y_pred=np.argmax(z,axis=1).astype(np.int64),probabilities=p)


def run_neural_job(X,P,L,F,job,paths,out,freeze,locksha,bank,pad,network,device,execution,stop_file=None,after_shard=None):
    import numpy as np
    m=job['model'];tr=L.load_development(paths['split'],m['seed'],m['context']['train_set']);va=L.load_development(paths['split'],m['seed'],m['context']['validation_set']);L.validate_pair(tr,va,m['context'])
    # Sentinel only, never choose numerical settings to improve a test score.
    vf=safe(paths['xlmr'],m['validation_folder']);vr=read(vf/'validation_report.json')
    need(sha(vf/'validation_report.json')==m['validation_report_sha256'],'Saved validation report changed.')
    vd=read(vf/'shard_00000.json');Verifier().check(vf/'shard_00000.PRIVATE.npz',vd)
    with np.load(vf/'shard_00000.PRIVATE.npz',allow_pickle=False) as a:
        n=min(32,len(va.targets));need(np.array_equal(a['row_ids'][:n],va.row_ids[:n]),'XLM saved validation rows differ.')
        expected_logits=a['logits'][:n];expected_pred=expected_logits.argmax(axis=1)
    tokens=P.aligned_tokens(bank,va,np.arange(n,dtype=np.int64));ids,mask=P.padded(tokens['token_ids'],pad,256)
    actual=logit_predict(network,ids,mask,device)
    need(np.allclose(actual['logits'],expected_logits,rtol=1e-4,atol=1e-4) and np.array_equal(actual['y_pred'],expected_pred),'XLM saved-validation sentinel differs; no automatic precision/batch change.')
    efmap={e['recipe']['experiment_id']:e for e in job['evaluations']}
    for e in sorted(job['evaluations'],key=lambda e:bool(e['recipe']['reuse_fitted_model_from'])):
        check_stop(stop_file);spec=prediction_spec(locksha,m,e);folder=safe(out/'evaluations',e['evaluation_key'])
        if ready(folder,digest(canonical(spec))):continue
        part=load_test(L,paths['split'],m['seed'],e['recipe']['test_set'],freeze);check_disjoint(part,tr,va)
        reuse=e['recipe']['reuse_fitted_model_from']
        if reuse:
            parent=efmap[reuse];payload=read_predictions(safe(out/'evaluations',parent['evaluation_key']),digest(canonical(prediction_spec(locksha,m,parent))))
            predict=extreme_predictor(payload,part);exe=dict(execution,reused_prediction_recipe=reuse)
        else:
            data=P.aligned_tokens(bank,part,np.arange(len(part),dtype=np.int64));ids,mask=P.padded(data['token_ids'],pad,256)
            def predict(lo,hi):return logit_predict(network,ids[lo:hi],mask[lo:hi],device)|{'token_lengths':data['token_lengths'][lo:hi]}
            exe=execution
        evaluate_set(L,folder,spec,m,part,predict,exe,stop_file,after_shard)
    return dict(state='SELECTED_MODEL_TESTS_READY',model_key=m['model_key'],validation_sentinel_passed=True)


def child(planfile):
    plan=read(planfile);out=Path(plan['out']);m=plan['job']['model'];statusdir=safe(out/'workers',m['model_key'])
    try:
        X=load_trainer(plan['project']);X.configure_cpu(4);P=X.load_pilot(plan['project']);L=P.load_helper(plan['project'],'linear');F=P.load_helper(plan['project'],'features')
        need(sha(__file__)==plan['code_sha256'],'Evaluation script changed during run.')
        need(sha(out/'selection_lock.json')==plan['locksha'],'Selection lock changed.')
        frozen=read(out/'selection_lock.json');matching=[q for q in frozen['models'] if q['model_key']==m['model_key']]
        need(len(matching)==1 and matching[0]==m,'Unfrozen/modified selected model requested.')
        import torch,numpy as np
        from transformers import AutoConfig,AutoModelForSequenceClassification
        need(F.versions()==plan['packages'],'Neural package versions changed.')
        need(torch.cuda.is_available() and torch.cuda.device_count()==1,'Exactly one authorized CUDA device required.')
        torch.cuda.set_device(0);torch.set_num_threads(4)
        actual=str(getattr(torch.cuda.get_device_properties(0),'uuid',''))
        need(actual.lower().removeprefix('gpu-')==plan['gpu']['uuid'].lower().removeprefix('gpu-'),'Wrong physical GPU.')
        torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        paths={k:Path(v) for k,v in plan['paths'].items()}
        snapshot,asset=F.verify_asset(plan['xlmr_stage'],Path(plan['hub_root']),'xlmr_tokens');need(asset['file_manifest_sha256']==plan['asset_sha256'],'Pinned base config/tokenizer changed.')
        names=['negative','neutral','positive'] if m['task']=='three_class' else ['negative','positive']
        config=AutoConfig.from_pretrained(str(snapshot),num_labels=len(names),id2label=dict(enumerate(names)),label2id={x:i for i,x in enumerate(names)},local_files_only=True,trust_remote_code=False)
        network=AutoModelForSequenceClassification.from_config(config,attn_implementation='eager').float()
        weights=safe(paths['xlmr'],m['model_file']);Verifier().check(weights,m['model_descriptor'])
        state=torch.load(weights,map_location='cpu',weights_only=True)
        need(state['signature']==m['selection']['selected_epoch_signature'],'Wrong selected epoch loaded.')
        network.load_state_dict(state['model'],strict=True);del state
        network.requires_grad_(False);network.eval();network.to('cuda:0')
        bank=F.FeatureBank(safe(paths['features'],plan['token_directory']));need(digest(canonical(bank.spec))==plan['token_spec_sha256'],'Token bank identity differs.')
        execution=dict(device='CUDA',gpu=plan['gpu'],precision='fp32',microbatch=4,attention='eager',TF32=False)
        result=run_neural_job(X,P,L,F,plan['job'],paths,out,frozen,plan['locksha'],bank,config.pad_token_id,network,'cuda:0',execution,plan.get('stop_file'))
        need(sha(weights)==m['model_descriptor']['sha256'],'Source selected weights changed.')
        write(statusdir/'status.json',result);return 0
    except BaseException as e:
        interrupted=_STOP or isinstance(e,(Stopped,KeyboardInterrupt))
        safe_message=str(e) if isinstance(e,EvaluationError) else 'Runtime failure; original exception retained in local PRIVATE log.'
        if type(e).__name__ in ('AcceleratorError','OutOfMemoryError'):
            import re
            match=re.search(r'CUDA (?:error|out of memory)[^\n]*',str(e))
            if match:safe_message=match.group(0)[:500]
        write(statusdir/'status.json',dict(state='INTERRUPTED' if interrupted else 'TEST_EVALUATION_BLOCKED',error_type=type(e).__name__,message=safe_message))
        atomic(statusdir/f'error_{time.time_ns()}.PRIVATE.log',lambda f:f.write(traceback.format_exc()),False)
        return 130 if interrupted else 1

def launch(planfile,gpu,stop_file=None):
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['HF_HUB_OFFLINE']='1';env['TRANSFORMERS_OFFLINE']='1'
    log=Path(planfile).parent/f'console_{time.time_ns()}.PRIVATE.log'
    with log.open('a',encoding='utf-8') as f:
        proc=subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),'--child-plan',str(planfile)],env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        offset=0;buf='';sent=False
        while True:
            if (_STOP or (stop_file and Path(stop_file).exists())) and not sent and proc.poll() is None:proc.send_signal(signal.SIGTERM);sent=True
            with log.open(encoding='utf-8',errors='replace') as q:q.seek(offset);buf+=q.read();offset=q.tell()
            lines=buf.split('\n');buf=lines.pop()
            for s in lines:
                if s.startswith(('[TEST]','[REUSE]')):print(s,flush=True)
            if proc.poll() is not None:break
            time.sleep(.4)
    if sent or proc.returncode==130:raise Stopped('Child stopped; completed prediction shards retained.')
    need(proc.returncode==0,'XLM inference child did not complete; inspect shareable status before PRIVATE logs.')


def summarize_reports(reports,expected_seeds=SEEDS):
    import statistics
    groups=defaultdict(list)
    for r in reports:groups[(r['experiment_id'],r['representation'],r['classifier'],r['task'])].append(r)
    out=[]
    for key,rs in sorted(groups.items()):
        need(len({r['seed'] for r in rs})==len(rs),'Duplicate seed result in aggregation.')
        row=dict(zip(('experiment_id','representation','classifier','task'),key),n_seeds=len(rs),complete_five_seed=set(r['seed'] for r in rs)==set(expected_seeds))
        for metric in ('macro_f1','accuracy','weighted_f1'):
            values=[r['metrics'][metric] for r in rs];row[metric+'_mean']=statistics.mean(values);row[metric+'_sd']=statistics.stdev(values) if len(values)>1 else None
        out.append(row)
    return out

def package(out,plan,locksha,state,message=None):
    reports=[];pending=[]
    for j in plan:
        for e in j['evaluations']:
            folder=safe(out/'evaluations',e['evaluation_key']);sig=digest(canonical(prediction_spec(locksha,j['model'],e)))
            if (folder/'complete.json').exists():
                # Full payload verification is performed by evaluation/reuse. Summary itself
                # checks committed report bytes without rereading every large prediction shard.
                d=read(folder/'complete.json');need(d['signature']==sig,'Summary found a mismatched evaluation.')
                Verifier().check(folder/'evaluation_report.json',d['files']['evaluation_report.json']);r=read(folder/'evaluation_report.json');need(r['signature']==sig,'Summary report signature differs.');reports.append(r)
            else:pending.append(e['evaluation_key'])
    full=len(reports)==sum(len(j['evaluations']) for j in plan)
    if state=='FINISHED':state='FINAL_TEST_RESULTS_READY' if full else 'REQUESTED_TEST_RESULTS_READY'
    r=dict(schema_version=VERSION,state=state,protocol_id=read(out/'root_spec.json')['protocol_id'],selection_lock_sha256=locksha,
        expected_evaluations=sum(len(j['evaluations']) for j in plan),completed_evaluations=len(reports),pending_evaluations=pending,
        final_test_evaluated=bool(reports),training_performed=False,model_selection_on_test=False,confidence_intervals_or_significance_in_this_stage=False,
        results=reports,finished_at=now())
    if message:r['message']=message
    workers=[]
    for p in (out/'workers').glob('seed_*/*/*/status.json') if (out/'workers').exists() else []:
        s=read(p)
        worker_key=str(p.parent.relative_to(out/'workers'))
        unfinished=any(k.startswith(worker_key+'/') for k in pending)
        if unfinished and s['state'] not in ('SELECTED_MODEL_TESTS_READY',):workers.append(dict(worker=worker_key,**s))
    r['worker_statuses']=workers
    write(out/'test_evaluation_report.json',r)
    rows=[];classes=[]
    for x in reports:
        row={k:x[k] for k in ('seed','experiment_id','context_id','representation','classifier','task','test_set','test_manifest_sha256','model_sha256','selected_C','selected_lr','selected_epoch')}
        rows.append(row|{k:x['metrics'][k] for k in ('n','accuracy','macro_f1','weighted_f1')}|dict(prediction_seconds=x['prediction_seconds']))
        for label in x['metrics']['class_order']:classes.append(row|dict(label=label,**x['metrics']['class_report'][label]))
    cs=['seed','experiment_id','context_id','representation','classifier','task','test_set','test_manifest_sha256','model_sha256','selected_C','selected_lr','selected_epoch']
    csvwrite(out/'all_test_results.csv',rows,cs+['n','accuracy','macro_f1','weighted_f1','prediction_seconds'])
    csvwrite(out/'class_test_results.csv',classes,cs+['label','precision','recall','f1-score','support'])
    summary=summarize_reports(reports)
    csvwrite(out/'test_mean_sd.csv',summary,['experiment_id','representation','classifier','task','n_seeds','complete_five_seed','macro_f1_mean','macro_f1_sd','accuracy_mean','accuracy_sd','weighted_f1_mean','weighted_f1_sd'])
    lines=['SAUCIS — SABITLENMIS MODELLERIN FINAL TEST DEGERLENDIRMESI','='*65,'Durum: '+state,
        f'Tamamlanan degerlendirme: {len(reports)}/{r["expected_evaluations"]}',
        'Egitim/refit/kalibrasyon/teste gore model secimi YOK. Test etiketleri yalnizca metriklerde kullanildi.',
        'LR/SVM ayri; XLM source-validation secimi sabit. Uc-puan gorunumleri binary tahminlerden alindi.',
        'Ortalama/SD betimsel; eslestirilmis guven araligi/anlamlilik analizi sonraki asama.',
        'PRIVATE tahminler, yorumlar ve ornek kimlikleri paylasim ZIPinde YOK.']
    if message:lines.append('Bilgi: '+message)
    atomic(out/'test_evaluation_summary.txt',lambda f:f.write('\n'.join(lines)+'\n'),False)
    public=['root_spec.json','test_evaluation_report.json','test_evaluation_summary.txt','all_test_results.csv','class_test_results.csv','test_mean_sd.csv','selection_lock_summary.json']
    def zwrite(f):
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            for n in public:z.write(out/n,n)
    atomic(out/'shareable_test_evaluation.zip',zwrite)
    print('\n'.join(lines),flush=True);print('[SHARE]',out/'shareable_test_evaluation.zip',flush=True)
    return r


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.')
    for k in ('split','features','smoke','linear','pilot','xlmr'):p.add_argument('--'+k)
    p.add_argument('--gpu',default='1',help='Only an explicitly authorized physical index/UUID; no auto-switch.')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--models',choices=('all','linear','xlmr'),default='all')
    p.add_argument('--plan-only',action='store_true',help='Lock all selected models and evaluation recipes WITHOUT reading TEST examples.')
    p.add_argument('--stop-file');p.add_argument('--hub-cache');p.add_argument('--child-plan',help=argparse.SUPPRESS)
    a=p.parse_args();signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    if a.child_plan:return child(a.child_plan)
    need(a.threads==4,'This reviewed evaluation uses four CPU threads; no silent execution-setting changes.')
    project=Path(a.project).expanduser().resolve();X=load_trainer(project);X.configure_cpu(4)
    need(all(getattr(a,k) for k in ('split','features','smoke','linear','pilot','xlmr')),'Provide all source run directories.')
    resolve=lambda s:(Path(s).expanduser() if Path(s).expanduser().is_absolute() else project/Path(s)).resolve()
    paths={k:resolve(getattr(a,k)) for k in ('split','features','smoke','linear','pilot','xlmr')}
    stopfile=str(resolve(a.stop_file)) if a.stop_file else None
    with ExitStack() as stack:
        stack.enter_context(lock(paths['linear']/'.lock',True));stack.enter_context(lock(paths['xlmr']/'.lock',True))
        print('[AUDIT] Selected models/validation rules and source checksums; no TEST rows yet.',flush=True)
        P,L,F,S,base,fr,contexts,asset,npkgs,pr,pad,tokenbank=X.inspect_environment(project,paths['split'],paths['features'],paths['smoke'],paths['linear'],paths['pilot'],a.hub_cache)
        v=Verifier();linear=audit_linear_models(L,P,paths['linear'],base,contexts,fr,paths['split'],v)
        neural,xr,xspec=audit_xlmr_reports(X,paths['xlmr'],base,contexts,v)
        need(xspec['linear_report_sha256']==sha(paths['linear']/'linear_training_report.json'),'XLM used a different linear audit.')
        need(xspec['neural_packages']==npkgs,'Evaluation neural environment differs from training.')
        models=linear+neural
        recipes={seed:read(paths['split']/f'seed_{seed}/experiments.json') for seed in SEEDS}
        plan=build_plan(models,recipes,lambda seed,sid:metadata(L,paths['split'],seed,sid))
        spec=dict(schema_version=VERSION,protocol_id=base['protocol_id'],runner_sha256=sha(__file__),trainer_sha256=TRAINER[1],
            corpus_sha256=base['corpus_sha256'],split_report_sha256=base['split_report_sha256'],feature_report_sha256=base['feature_report_sha256'],
            linear_report_sha256=sha(paths['linear']/'linear_training_report.json'),xlmr_report_sha256=sha(paths['xlmr']/'xlmr_training_report.json'),
            linear_packages=base['packages'],neural_packages=npkgs,settings=SETTINGS,model_registry_sha256=digest(canonical(models)),plan_sha256=digest(canonical(plan)))
        out=project/'_revision_test'/f'test_{base["protocol_id"]}_{spec["runner_sha256"][:10]}'
        stack.enter_context(lock(out/'.lock'));print('[OUTPUT]',out,flush=True)
        if (out/'root_spec.json').exists():need(read(out/'root_spec.json')==spec,'Existing TEST run has changed inputs/settings; no overwrite.')
        else:write(out/'root_spec.json',spec)
        frozen=dict(state='SELECTIONS_LOCKED_BEFORE_TEST',model_count=len(models),models=models,evaluation_count=2820,plan_sha256=spec['plan_sha256'])
        if (out/'selection_lock.json').exists():need(read(out/'selection_lock.json')==frozen,'Frozen selection registry changed.')
        else:write(out/'selection_lock.json',frozen)
        if (out/'evaluation_plan.json').exists():need(read(out/'evaluation_plan.json')==plan,'Frozen TEST recipe plan changed.')
        else:write(out/'evaluation_plan.json',plan)
        locksha=sha(out/'selection_lock.json')
        write(out/'selection_lock_summary.json',dict(state=frozen['state'],linear_models=len(linear),xlmr_models=len(neural),linear_evaluations=2700,xlmr_evaluations=120,selection_lock_sha256=locksha,
            test_rows_read_to_select_models=False,weight_verification='All selected models/transformers and selected validation payloads checked; unselected epoch weights not reread.'))
        print('[LOCKED] 1200 linear + 45 XLM-R choices; 2820 evaluation cells.',flush=True)
        if a.plan_only:package(out,plan,locksha,'TEST_PLAN_LOCKED');return 0
        rc=0;state='FINISHED';message=None
        try:
            if a.models in ('all','linear'):run_linear(X,L,plan,paths,out,frozen,locksha,fr,base,stopfile)
            if a.models in ('all','xlmr'):
                gpu=None
                for j in (j for j in plan if j['model']['kind']=='xlmr'):
                    check_stop(stopfile)
                    if all(ready(safe(out/'evaluations',e['evaluation_key']),digest(canonical(prediction_spec(locksha,j['model'],e)))) for e in j['evaluations']):
                        print('[REUSE]',j['model']['model_key'],'verified; no GPU inference',flush=True);continue
                    if gpu is None:gpu=S.safe_gpu_info(a.gpu);print('[GPU]',gpu['index'],gpu['uuid'],gpu['name'],flush=True)
                    worker=safe(out/'workers',j['model']['model_key']);worker.mkdir(parents=True,exist_ok=True)
                    cp=dict(project=str(project),out=str(out),paths={k:str(v) for k,v in paths.items()},job=j,locksha=locksha,code_sha256=sha(__file__),gpu=gpu,packages=npkgs,
                        hub_root=str(F.hub_root(a.hub_cache)),xlmr_stage={s['stage']:s for s in read(paths['smoke']/'model_smoke_report.json')['stages']}['xlmr'],
                        asset_sha256=asset['file_manifest_sha256'],token_directory=fr['banks']['xlmr_tokens']['bank_directory'],token_spec_sha256=fr['banks']['xlmr_tokens']['spec_sha256'],stop_file=stopfile)
                    write(worker/'launch_plan.local.json',cp);print('[MODEL]',j['model']['model_key'],flush=True);launch(worker/'launch_plan.local.json',gpu,stopfile)
        except BaseException as e:
            state='INTERRUPTED' if _STOP or isinstance(e,(Stopped,KeyboardInterrupt)) else 'TEST_EVALUATION_BLOCKED';rc=130 if state=='INTERRUPTED' else 1
            message=str(e) if isinstance(e,EvaluationError) else 'Runtime/input error; original traceback is in the local PRIVATE log.'
            atomic(out/f'error_{time.time_ns()}.PRIVATE.log',lambda f:f.write(traceback.format_exc()),False)
        finally:package(out,plan,locksha,state,message)
        return rc

if __name__=='__main__':
    try:sys.exit(main())
    except EvaluationError as e:print('[BLOCKED]',str(e),file=sys.stderr);sys.exit(1)
