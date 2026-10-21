#!/usr/bin/env python3
"""SAUCIS step 05: full development-only TF-IDF/frozen-embedding classifier training.

Uses immutable step-02 memberships and step-04 feature banks. For each source
context, representation and classifier, all five C values are trained independently
and selected ONLY on the recipe's permitted validation set. No final-test rows or
labels are fetched. No encoder, CUDA, network or package installation is used.

Completed individual C fits and transformers are checksum-protected and resumed.
An interrupted active sklearn fit is restarted; completed fits are not repeated.
All candidate models are retained, not just the winner. Joblib files are loaded
only from this runner's own checksum-matched output; never load untrusted files.
"""
from __future__ import annotations
import argparse
from collections import Counter
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
import sys
import time
import traceback
import warnings
import zipfile

VERSION = '1.0.0'
C_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
REPRESENTATIONS = ('word', 'char', 'minilm', 'labse', 'bge_m3')
CLASSIFIERS = ('lr', 'svm')
DIMENSIONS = {'minilm': 384, 'labse': 768, 'bge_m3': 1024}
LIMITS = {'minilm': 128, 'labse': 256, 'bge_m3': 8192, 'xlmr_tokens': 256}
LIBRARIES = ('numpy', 'scipy', 'scikit-learn', 'joblib', 'threadpoolctl')

class TrainingError(RuntimeError):
    """Only aggregate/safe diagnostics in these exception messages."""

class Stopped(TrainingError): pass

def need(ok, message):
    if not ok: raise TrainingError(message)

def now(): return datetime.now(timezone.utc).isoformat()
def canonical(x): return json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
def digest(x): return hashlib.sha256(x.encode('utf-8')).hexdigest()
def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()
def load_json(path): return json.loads(Path(path).read_text(encoding='utf-8'))

def atomic_file(path, writer, binary=True):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f'.partial_{os.getpid()}_{time.time_ns()}')
    kwargs = {} if binary else {'encoding': 'utf-8'}
    with tmp.open('wb' if binary else 'w', **kwargs) as f:
        writer(f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, path)

def write_json(path, obj):
    atomic_file(path, lambda f: f.write(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n'), False)

def write_npz(path, arrays):
    import numpy as np
    atomic_file(path, lambda f: np.savez(f, **arrays))

def write_model(path, model):
    import joblib
    atomic_file(path, lambda f: joblib.dump(model, f, compress=3))

def safe_path(root, relative):
    r = Path(relative)
    need(not r.is_absolute() and '..' not in r.parts, 'Unsafe relative artifact path.')
    p = Path(root)/r
    need(p.resolve().is_relative_to(Path(root).resolve()), 'Artifact escapes its run directory.')
    return p

@contextmanager
def ro_db(path):
    p = Path(path).resolve(); need(p.is_file(), 'Required local SQLite file is missing.')
    db = sqlite3.connect(p.as_uri()+'?mode=ro', uri=True)
    db.execute('PRAGMA query_only=ON')
    try: yield db
    finally: db.close()

@contextmanager
def lock(path):
    import fcntl
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open('a') as f:
        try: fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as e: raise TrainingError('This output is already being used. Do not run concurrent writers.') from e
        try: yield
        finally: fcntl.flock(f, fcntl.LOCK_UN)

def versions():
    out = {'python': platform.python_version()}
    for name in LIBRARIES:
        try: out[name] = md.version(name)
        except md.PackageNotFoundError: out[name] = None
    return out

def signal_stop(signum, frame):
    raise Stopped('Stopped. Completed C fits are retained; the active incomplete fit will restart.')

def check_stop(path=None):
    if path and Path(path).exists(): raise Stopped('Stop file found; remove it explicitly before resuming.')


def complete_bundle(folder, spec, filenames):
    outputs = {}
    for name in filenames:
        p = safe_path(folder, name)
        need(p.is_file(), 'Cannot complete a bundle with a missing file.')
        outputs[name] = {'sha256': file_sha(p), 'bytes': p.stat().st_size}
    stamp = {'state': 'COMPLETE', 'identity_sha256': digest(canonical(spec)), 'outputs': outputs}
    write_json(Path(folder)/'complete.json', stamp)
    return stamp


def verified_bundle(folder, spec):
    folder = Path(folder); p = folder/'complete.json'
    if not p.exists(): return None
    stamp = load_json(p)
    need(stamp['state']=='COMPLETE' and stamp['identity_sha256']==digest(canonical(spec)),
         'Completed output identity differs; do not silently reuse or overwrite it.')
    need(bool(stamp['outputs']), 'Empty completion record.')
    for name, item in stamp['outputs'].items():
        path = safe_path(folder, name)
        need(path.is_file() and path.stat().st_size==item['bytes'] and file_sha(path)==item['sha256'],
             'Completed artifact is missing/corrupt; no automatic deletion or recomputation.')
    return stamp


@dataclass
class Development:
    set_id: str
    role: str
    task: str
    row_ids: object
    sample_ids: object
    group_ids: object
    texts: list
    targets: object
    languages: object
    domains: object
    text_hashes: object
    metadata: dict

    def identity(self):
        return {'set_id': self.set_id, 'role': self.role, 'task': self.task,
                'rows': len(self.targets), 'manifest_sha256': self.metadata['manifest_sha256']}


def load_development(split, seed, set_id):
    """The only labeled data reader in this script: refuses test roles BEFORE joining records."""
    import numpy as np
    split = Path(split)
    with ro_db(split/f'seed_{seed}'/'memberships.local.sqlite3') as db:
        meta = db.execute('SELECT role,task,n_rows,metadata_json FROM sets WHERE set_id=?', (set_id,)).fetchone()
        need(meta is not None, 'Unknown development set.')
        role, task, n, raw = meta
        need(role in ('train','validation'), 'Development reader refuses final-test data.')
        need(task in ('three_class','binary'), 'Unknown label schema.')
        info = json.loads(raw)
        db.execute('ATTACH DATABASE ? AS corpus', ((split/'corpus.local.sqlite3').resolve().as_uri()+'?mode=ro',))
        rows = db.execute('''SELECT r.row_id,r.sample_id,r.group_id,r.model_text,m.target,
            r.language,r.domain,r.label FROM members m JOIN corpus.records r USING(row_id)
            WHERE m.set_id=? ORDER BY r.row_id''', (set_id,)).fetchall()
    need(len(rows)==n and n>0, 'Development row count differs from the manifest.')
    need(len({r[0] for r in rows})==n and len({r[1] for r in rows})==n, 'Duplicate development records.')
    h = hashlib.sha256()
    for r in rows:
        h.update((r[1]+':'+str(r[4])+'\n').encode())
        expected = r[7] if task=='three_class' else (0 if r[7]==0 else 1)
        need((task=='three_class' or r[7] in (0,2)) and expected==r[4], 'Task label mapping differs from corpus.')
    need(h.hexdigest()==info['manifest_sha256'], 'Development IDs/labels differ from the manifest.')
    need(dict(Counter(r[5] for r in rows))==info['language_counts'], 'Language counts differ.')
    return Development(set_id, role, task, np.array([r[0] for r in rows],dtype=np.int64),
        np.array([r[1] for r in rows],dtype='U64'), np.array([r[2] for r in rows],dtype='U64'),
        [r[3] for r in rows], np.array([r[4] for r in rows],dtype=np.int64),
        np.array([r[5] for r in rows]), np.array([r[6] for r in rows]),
        np.array([digest(r[3]) for r in rows],dtype='U64'), info)


def task_labels(task):
    need(task in ('three_class','binary'), 'Unknown task.')
    return [0,1,2] if task=='three_class' else [0,1]


def contexts_from_recipes(recipes):
    contexts = {}
    for e in recipes:
        need(e.get('test_labels_must_not_select_models') is True, 'Unsafe selection recipe.')
        c = {'context_id':e['training_context_id'], 'train_set':e['train_set'],
             'validation_set':e['validation_set'], 'selection_metric':e['selection_metric'],
             'validation_languages':sorted(e['validation_languages'])}
        key = c['context_id']
        if key in contexts:
            need(all(contexts[key][k]==v for k,v in c.items()), 'Shared model has inconsistent selection recipes.')
        else: contexts[key] = dict(c, evaluation_recipe_ids=[])
        contexts[key]['evaluation_recipe_ids'].append(e['experiment_id'])
    return contexts


def validate_pair(train, val, context):
    labs = set(task_labels(train.task))
    need(train.role=='train' and val.role=='validation' and train.task==val.task, 'Invalid development roles/tasks.')
    need(set(train.group_ids).isdisjoint(set(val.group_ids)), 'Train and validation text groups overlap.')
    langs = set(context['validation_languages'])
    need(set(train.languages)==langs and set(val.languages)==langs, 'Source-language selection violation.')
    need(context['selection_metric']==('equal_language_mean_macro_f1' if langs=={'tr','kz'} else 'source_macro_f1'),
         'Unexpected validation selection rule.')
    for part in (train,val):
        for lang in langs: need(set(part.targets[part.languages==lang])==labs, 'A source language is missing a class.')


def metric_pack(y, pred, labels):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
    y = np.asarray(y); pred = np.asarray(pred)
    need(y.ndim==pred.ndim==1 and len(y)>0 and len(y)==len(pred), 'Invalid metric arrays.')
    need(set(y)<=set(labels) and set(pred)<=set(labels), 'Metric label outside task.')
    names = ['negative','neutral','positive'] if labels==[0,1,2] else ['negative','positive']
    return {'n':len(y), 'accuracy':float(accuracy_score(y,pred)),
        'macro_f1':float(f1_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'weighted_f1':float(f1_score(y,pred,labels=labels,average='weighted',zero_division=0)),
        'macro_precision':float(precision_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'macro_recall':float(recall_score(y,pred,labels=labels,average='macro',zero_division=0)),
        'class_report':classification_report(y,pred,labels=labels,target_names=names,output_dict=True,zero_division=0),
        'class_order':names, 'confusion_matrix':confusion_matrix(y,pred,labels=labels).tolist()}


def validation_metrics(val, predictions, context):
    labels = task_labels(val.task)
    perlang = {lang:metric_pack(val.targets[val.languages==lang], predictions[val.languages==lang], labels)
               for lang in context['validation_languages']}
    # Exactly equal language weight, NOT pooled F1 or a row-count weighted mean.
    score = sum(v['macro_f1'] for v in perlang.values())/len(perlang)
    need(math.isfinite(score), 'Nonfinite validation selection score.')
    return {'selection_score':score, 'selection_rule':context['selection_metric'],
            'by_language':perlang, 'pooled':metric_pack(val.targets,predictions,labels)}


def vectorizer(representation):
    from sklearn.feature_extraction.text import TfidfVectorizer
    need(representation in ('word','char'), 'Unknown lexical representation.')
    return TfidfVectorizer(analyzer=representation, ngram_range=(1,2) if representation=='word' else (3,5),
        max_features=80000 if representation=='word' else 120000, min_df=2, lowercase=True,
        sublinear_tf=True, smooth_idf=True, norm='l2', strip_accents=None)


def classifier(kind, C, seed):
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    if kind=='lr':
        return LogisticRegression(C=C, solver='lbfgs', max_iter=3000, tol=1e-4,
                                  class_weight='balanced', random_state=seed)
    need(kind=='svm','Unknown classifier.')
    return LinearSVC(C=C, dual='auto', penalty='l2', loss='squared_hinge', multi_class='ovr',
                     max_iter=10000, tol=1e-4, class_weight='balanced', random_state=seed)


def serial_params(estimator):
    def convert(x):
        if x is None or isinstance(x,(str,bool,int,float)): return x
        if isinstance(x,(list,tuple)): return [convert(v) for v in x]
        if isinstance(x,dict): return {str(k):convert(v) for k,v in x.items()}
        return str(x)
    return convert(estimator.get_params(deep=False))


def vector_signature(v):
    import numpy as np
    return digest(canonical({'vocabulary':sorted((k,int(i)) for k,i in v.vocabulary_.items()),
                             'idf_sha256':hashlib.sha256(np.asarray(v.idf_).tobytes()).hexdigest(),
                             'parameters':serial_params(v)}))


def matrix_stats(X):
    import numpy as np
    from scipy.sparse import issparse
    if issparse(X):
        zero = np.diff(X.indptr)==0; finite = np.isfinite(X.data).all(); nnz = X.nnz
    else:
        zero = np.count_nonzero(X,axis=1)==0; finite=np.isfinite(X).all(); nnz=int(np.count_nonzero(X))
    need(bool(finite), 'Nonfinite model features.')
    return {'shape':list(X.shape), 'dtype':str(X.dtype), 'nonzero_entries':int(nnz),
            'zero_rows':int(zero.sum()), 'zero_fraction':float(zero.mean())}


class FrozenBank:
    """Loads one immutable full label-free bank at a time; validates every shard.

    Full-bank reads do not read corpus labels/test memberships. Development subset
    extraction below separately checks sample IDs and exact model-text hashes.
    """
    def __init__(self, folder, expected, corpus_sha, protocol):
        import numpy as np
        self.folder=Path(folder);self.spec=load_json(self.folder/'bank_spec.json')
        done=load_json(self.folder/'bank.complete.json');self.report=load_json(self.folder/'bank_report.json')
        s=self.spec; h=digest(canonical(s))
        need(s['kind']=='frozen_embeddings' and s['alias'] in DIMENSIONS,'Not a frozen embedding bank.')
        need(s['dimension']==DIMENSIONS[s['alias']] and s['max_length']==LIMITS[s['alias']], 'Encoder settings changed.')
        need(s['corpus_sha256']==corpus_sha and s['protocol_id']==protocol,'Feature/corpus/split identity mismatch.')
        need(s['labels_read'] is False and s['fit_on_corpus'] is False and s['normalize_embeddings'] is True,
             'Unsupported fitted or unnormalized bank.')
        need(done['spec_sha256']==h==expected['spec_sha256'], 'Feature spec hash mismatch.')
        need(done['report_sha256']==file_sha(self.folder/'bank_report.json'), 'Feature report checksum mismatch.')
        need(self.report['state']=='FEATURE_BANK_READY' and self.report['rows']==s['rows']==expected['rows'], 'Incomplete feature bank.')
        n=s['rows']; self.E=np.empty((n,s['dimension']),np.float32)
        self.sample_ids=np.empty(n,dtype='S64');self.text_hashes=np.empty(n,dtype='S64')
        self.lengths=np.empty(n,np.int32);first=1
        for i,d in enumerate(self.report['shard_files']):
            data=safe_path(self.folder,d['file']);meta=safe_path(self.folder,d['metadata_file'])
            need(file_sha(data)==d['sha256'] and file_sha(meta)==d['metadata_sha256'],'Corrupt feature shard.')
            m=load_json(meta)
            need(m['spec_sha256']==h and m['sha256']==d['sha256'] and m['first_row_id']==d['first_row_id']==first,
                 'Feature shard identity or ordering differs.')
            with np.load(data,allow_pickle=False) as a:
                need(set(a.files)=={'row_ids','sample_ids','text_sha256','token_lengths','embeddings'}, 'Unexpected embedding payload.')
                count=d['rows'];last=first+count-1
                need(d['last_row_id']==last and np.array_equal(a['row_ids'],np.arange(first,last+1)), 'Feature row alignment error.')
                need(a['embeddings'].dtype==np.float32 and a['embeddings'].shape==(count,s['dimension']), 'Embedding shape/dtype mismatch.')
                need(np.isfinite(a['embeddings']).all() and np.allclose(np.linalg.norm(a['embeddings'],axis=1),1,atol=2e-4),
                     'Invalid embedding values or normalization.')
                for key in ('sample_ids','text_sha256'):
                    need(a[key].dtype==np.dtype('S64') and a[key].shape==(count,), 'Invalid feature identity vector.')
                need(a['token_lengths'].shape==(count,) and np.all(a['token_lengths']>0),'Invalid token lengths.')
                sl=slice(first-1,last);self.E[sl]=a['embeddings'];self.sample_ids[sl]=a['sample_ids']
                self.text_hashes[sl]=a['text_sha256'];self.lengths[sl]=a['token_lengths']
            first=last+1
        need(first==n+1 and len(np.unique(self.sample_ids))==n, 'Feature bank coverage/IDs invalid.')

    def development(self, part):
        import numpy as np
        ids=part.row_ids
        need(len(ids)>0 and np.all(ids>=1) and np.all(ids<=len(self.E)), 'Requested feature rows out of range.')
        ix=ids-1
        need(np.array_equal(self.sample_ids[ix],part.sample_ids.astype('S64')), 'Feature sample identities do not match development set.')
        need(np.array_equal(self.text_hashes[ix],part.text_hashes.astype('S64')), 'Feature text hashes do not match development set.')
        return np.ascontiguousarray(self.E[ix])


def inspect_inputs(split, features, smoke):
    split=Path(split);features=Path(features);smoke=Path(smoke)
    protocol=load_json(split/'protocol.json');report=load_json(split/'split_report.json')
    need(report['state']=='SPLIT_MANIFESTS_READY' and report['blocked_checks']==0,'Split manifests are not ready.')
    need(protocol['protocol_id']==report['protocol_id'],'Split protocol identity mismatch.')
    need(load_json(split/'status.json')['report_sha256']==file_sha(split/'split_report.json'), 'Split report checksum mismatch.')
    cs=load_json(split/'corpus.complete.json')['sha256']
    need(cs==file_sha(split/'corpus.local.sqlite3'),'Corpus has changed.')
    with ro_db(split/'corpus.local.sqlite3') as db:
        need(db.execute('PRAGMA quick_check').fetchone()[0]=='ok','Corpus integrity error.')
        n=db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
    sr=load_json(smoke/'model_smoke_report.json')
    need(sr['state']=='MODEL_SMOKE_PASSED' and sr['protocol_id']==report['protocol_id'], 'Matching successful smoke report required.')
    need({s['stage'] for s in sr['stages'] if s['state']=='PASS'}=={'cpu','cuda','minilm','labse','bge_m3','xlmr'},'Incomplete smoke stages.')
    fr=load_json(features/'feature_build_report.json')
    need(fr['state']=='FEATURE_BANKS_READY' and fr['protocol_id']==report['protocol_id'], 'Complete matching feature report required.')
    need(fr['corpus_sha256']==cs and fr['input_rows']==n,'Feature report refers to a different corpus.')
    need(fr['smoke_report_sha256']==file_sha(smoke/'model_smoke_report.json'), 'Feature/smoke report identities differ.')
    env=versions()
    for name in ('python','numpy','scipy','scikit-learn','joblib'):
        need(env[name] is not None and env[name]==sr['packages'].get(name),
             f'Package {name} differs from smoke; no silent environment upgrade allowed.')
    need(env['threadpoolctl'] is not None,'threadpoolctl missing; do not auto-install.')
    need(fr['packages']['numpy']==env['numpy'],'Feature and training NumPy versions differ.')
    allcontexts={};seeds=report['seeds']
    for seed in seeds:
        root=split/f'seed_{seed}';done=load_json(root/'complete.json')
        for name,key in (('memberships.local.sqlite3','db_sha256'),('seed_report.json','report_sha256'),('experiments.json','recipes_sha256')):
            need(file_sha(root/name)==done[key], f'Seed {seed} manifest checksum mismatch.')
        r=load_json(root/'seed_report.json')
        need(r['state']=='SPLIT_MANIFESTS_READY' and not r['errors'] and all(c['status']=='PASS' for c in r['checks']),
             'A seed has blocked checks.')
        allcontexts[seed]=contexts_from_recipes(load_json(root/'experiments.json'))
    return {'protocol_id':report['protocol_id'],'corpus_sha256':cs,'rows':n,'packages':env,
            'split_report_sha256':file_sha(split/'split_report.json'),
            'feature_report_sha256':file_sha(features/'feature_build_report.json'),
            'smoke_report_sha256':file_sha(smoke/'model_smoke_report.json')}, fr, allcontexts


def safe_name(name): return re.sub('[^a-zA-Z0-9_.-]','_',name)+'_'+digest(name)[:8]

def write_development_ids(path, tr, va):
    write_npz(path, {'train_row_ids':tr.row_ids,'train_sample_ids':tr.sample_ids.astype('S64'),
                     'train_group_ids':tr.group_ids.astype('S64'),'train_targets':tr.targets,
                     'validation_row_ids':va.row_ids,'validation_sample_ids':va.sample_ids.astype('S64'),
                     'validation_group_ids':va.group_ids.astype('S64'),'validation_targets':va.targets})


def lexical_matrices(folder, representation, train, val, spec):
    import joblib
    folder=Path(folder)/'transformer';identity={'run_spec':spec,'stage':'train_only_tfidf'}
    started=time.monotonic()
    if verified_bundle(folder,identity):
        v=joblib.load(folder/'vectorizer.PRIVATE.joblib')
        info=load_json(folder/'transformer_report.json')
        need(info['signature']==vector_signature(v), 'Reloaded transformer differs.')
        Xtr=v.transform(train.texts)
    else:
        v=vectorizer(representation);Xtr=v.fit_transform(train.texts)
        info={'stage':'TRAIN_ONLY_TFIDF','train_set':train.identity(),'parameters':serial_params(v),
              'vocabulary_size':len(v.vocabulary_), 'signature':vector_signature(v),
              'fit_seconds':time.monotonic()-started, 'fit_rows':len(train.targets),
              'validation_used_for_vocabulary_or_idf':False,'final_test_accessed':False}
        write_model(folder/'vectorizer.PRIVATE.joblib',v);write_json(folder/'transformer_report.json',info)
        complete_bundle(folder,identity,['vectorizer.PRIVATE.joblib','transformer_report.json'])
    before=vector_signature(v);Xva=v.transform(val.texts)
    need(before==vector_signature(v),'Validation transform modified vocabulary/IDF.')
    return Xtr, Xva, info


def winner(trials):
    need(len(trials)==len(C_GRID) and {float(t['C']) for t in trials}==set(C_GRID),'Selection requires all five C fits.')
    need(all(t['state']=='CANDIDATE_READY' and math.isfinite(t['validation']['selection_score']) for t in trials),
         'Cannot select from an incomplete/nonconverged search.')
    return min(trials,key=lambda t:(-t['validation']['selection_score'],float(t['C'])))


def fit_candidate(folder, spec, kind, C, seed, Xtr, train, Xva, val, context, execution, hook=None):
    import numpy as np
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    folder=Path(folder);identity={'run_spec':spec,'classifier':kind,'C':float(C)}
    if verified_bundle(folder,identity):
        r=load_json(folder/'candidate_report.json')
        need(r['identity_sha256']==digest(canonical(identity)) and r['state']=='CANDIDATE_READY','Invalid saved C fit.')
        return r,True
    model=classifier(kind,C,seed);begun=time.monotonic()
    folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'status.json',{'state':'FITTING','started_at':now(),'identity_sha256':digest(canonical(identity))})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(Xtr,train.targets)
    converged=not any(issubclass(w.category,ConvergenceWarning) for w in caught)
    if not converged:
        write_json(folder/'status.json',{'state':'CONVERGENCE_BLOCKED','C':C,'classifier':kind,
                  'n_iter':np.asarray(model.n_iter_).tolist(), 'max_iter':model.max_iter,
                  'note':'Not eligible for model selection. No automatic iteration/tolerance change.'})
        raise TrainingError('A classifier did not converge. Completed earlier fits are retained; inspect CONVERGENCE_BLOCKED status.')
    need(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(),'Nonfinite model parameters.')
    labels=task_labels(train.task);need(model.classes_.tolist()==labels,'Classifier class order mismatch.')
    fit_seconds=time.monotonic()-begun;startpred=time.monotonic()
    p=model.predict(Xva);decision=np.asarray(model.decision_function(Xva))
    need(np.isfinite(decision).all(),'Nonfinite decision values.')
    metrics=validation_metrics(val,p,context)
    payload={'row_ids':val.row_ids,'sample_ids':val.sample_ids.astype('S64'),'group_ids':val.group_ids.astype('S64'),
             'y_true':val.targets,'y_pred':p.astype(np.int64),'languages':val.languages,
             'domains':val.domains,'decision_scores':decision,'class_ids':model.classes_}
    if kind=='lr':
        proba=model.predict_proba(Xva);need(np.isfinite(proba).all() and np.allclose(proba.sum(axis=1),1),'Invalid class probabilities.')
        payload['probabilities']=proba
    # For SVM, decision_scores are not probabilities; no implicit calibration.
    prediction_seconds=time.monotonic()-startpred
    write_npz(folder/'validation_predictions.PRIVATE.npz',payload)
    write_model(folder/'classifier.PRIVATE.joblib',model)
    reloaded=joblib.load(folder/'classifier.PRIVATE.joblib')
    need(np.array_equal(reloaded.predict(Xva[:min(64,len(val.targets))]),p[:min(64,len(val.targets))]),'Model reload changed predictions.')
    counts=Counter(int(x) for x in train.targets);weights={str(k):len(train.targets)/(len(labels)*counts[k]) for k in labels}
    r={'state':'CANDIDATE_READY','identity_sha256':digest(canonical(identity)),'C':float(C),'classifier':kind,
       'seed':seed,'context_id':context['context_id'],'representation':spec['representation'],
       'task':train.task,'train_rows':len(train.targets),'validation_rows':len(val.targets),
       'train_class_counts':{str(k):v for k,v in sorted(counts.items())},'effective_class_weights':weights,
       'parameters':serial_params(model),'n_iter':np.asarray(model.n_iter_).tolist(),
       'warning_categories':[w.category.__name__ for w in caught], 'converged':True,
       'fit_seconds':fit_seconds,'validation_prediction_seconds':prediction_seconds,'execution':execution,
       'validation':metrics,'final_test_predictions_created':False,'created_at':now()}
    write_json(folder/'candidate_report.json',r)
    complete_bundle(folder,identity,['classifier.PRIVATE.joblib','validation_predictions.PRIVATE.npz','candidate_report.json'])
    write_json(folder/'status.json',{'state':'CANDIDATE_READY','completed_at':now()})
    if hook:hook(r)
    return r,False


def run_representation(folder, base_spec, representation, seed, context, train, val, bank, threads, stop_file=None, hook=None):
    folder=Path(folder)
    feature_identity=bank.report['spec_sha256'] if bank else None
    spec={'root_spec_sha256':digest(canonical(base_spec)), 'seed':seed, 'context':context,
          'train':train.identity(),'validation':val.identity(),'representation':representation,
          'feature_spec_sha256':feature_identity,'C_grid':list(C_GRID)}
    cached=verified_bundle(folder,spec)
    if cached:
        # Validate every retained candidate and transformer, not merely selection.json.
        if representation in ('word','char'):
            need(verified_bundle(folder/'transformer',{'run_spec':spec,'stage':'train_only_tfidf'}) is not None,'Missing fitted transformer.')
        for kind in CLASSIFIERS:
            trials=[]
            for i,C in enumerate(C_GRID):
                cdir=folder/kind/f'C_{i}'
                need(verified_bundle(cdir,{'run_spec':spec,'classifier':kind,'C':C}) is not None,'Missing completed C fit.')
                trials.append(load_json(cdir/'candidate_report.json'))
            selected=load_json(folder/kind/'selected_config.json')
            need(selected['selected_C']==winner(trials)['C'], 'Saved selection violates validation-only rule.')
        print(f'[REUSE] seed={seed} context={context["context_id"]} representation={representation}',flush=True)
        return load_json(folder/'representation_report.json'),True
    validate_pair(train,val,context);folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'run_spec.json',spec)
    write_development_ids(folder/'development_ids.PRIVATE.npz',train,val)
    if representation in ('word','char'):
        Xtr,Xva,transform=lexical_matrices(folder,representation,train,val,spec)
    else:
        need(bank is not None and bank.spec['alias']==representation,'Missing matching frozen bank.')
        Xtr,Xva=bank.development(train),bank.development(val)
        transform={'kind':'frozen_features','feature_spec_sha256':feature_identity,
                   'train_truncated':int((bank.lengths[train.row_ids-1]>bank.spec['max_length']).sum()),
                   'validation_truncated':int((bank.lengths[val.row_ids-1]>bank.spec['max_length']).sum())}
    stats={'train':matrix_stats(Xtr),'validation':matrix_stats(Xva)}
    selections=[];trial_index=[]
    execution={'threads_requested':threads,'cpu_affinity_count':len(os.sched_getaffinity(0)) if hasattr(os,'sched_getaffinity') else os.cpu_count(),
               'machine':platform.machine(),'processor':platform.processor(),'GPU_used':False}
    for kind in CLASSIFIERS:
        rows=[]
        for i,C in enumerate(C_GRID):
            check_stop(stop_file)
            print(f'[FIT] seed={seed} context={context["context_id"]} rep={representation} classifier={kind} C={C:g}',flush=True)
            r,reused=fit_candidate(folder/kind/f'C_{i}',spec,kind,C,seed,Xtr,train,Xva,val,context,execution,hook)
            rows.append(r);trial_index.append({'classifier':kind,'C':C,'validation_score':r['validation']['selection_score'],
                'converged':True,'fit_seconds':r['fit_seconds'],'candidate_relative_path':f'{kind}/C_{i}'})
        best=winner(rows);ix=list(C_GRID).index(best['C'])
        selected={'state':'VALIDATION_SELECTED','seed':seed,'context_id':context['context_id'],
            'representation':representation,'classifier':kind,'selected_C':best['C'],
            'selection_score':best['validation']['selection_score'],'selection_rule':context['selection_metric'],
            'validation_languages':context['validation_languages'],'tie_break':'smallest_C_on_exact_score_tie',
            'train_rows':len(train.targets),'validation_rows':len(val.targets),'task':train.task,
            'selected_model_relative_path':f'{kind}/C_{ix}/classifier.PRIVATE.joblib',
            'selected_model_sha256':file_sha(folder/kind/f'C_{ix}'/'classifier.PRIVATE.joblib'),
            'selected_predictions_relative_path':f'{kind}/C_{ix}/validation_predictions.PRIVATE.npz',
            'validation':best['validation'],'final_test_used_for_selection':False,
            'refit_train_plus_validation':False,'evaluation_recipe_ids':context['evaluation_recipe_ids']}
        write_json(folder/kind/'selected_config.json',selected);selections.append(selected)
        print(f'[SELECTED] {representation}+{kind}: C={best["C"]:g}; source validation={selected["selection_score"]:.6f}',flush=True)
    summary={'state':'REPRESENTATION_READY','seed':seed,'context_id':context['context_id'],'representation':representation,
             'selections':selections,'trials':trial_index,'feature_summary':transform,'matrix_stats':stats,
             'train':train.identity(),'validation':val.identity(),'final_test_predictions_created':False}
    write_json(folder/'representation_report.json',summary)
    complete_bundle(folder,spec,['run_spec.json','development_ids.PRIVATE.npz','representation_report.json',
                               'lr/selected_config.json','svm/selected_config.json'])
    return summary,False


def package_report(out, report):
    write_json(out/'linear_training_report.json',report)
    lines=['SAUCIS — VALIDATION ILE SECILEN DOGRUSAL MODELLER','='*65,'Durum: '+report['state'],
           'Protocol: '+report['protocol_id'],
           f"Tamamlanan temsil-context isleri: {len(report['completed'])}/{report['requested_representation_jobs']}",
           f"Secilmis classifier sayisi: {2*len(report['completed'])}; tamamlanan C-fit sayisi: {10*len(report['completed'])}",
           'Bu asama GERCEK egitimdir; final-test skor/tahmini ve XLM-R fine-tuning YOK.',
           'Test etiketleri okunmadi. LR/SVM testten secilmedi. Her biri ayri kaydedildi.',
           'Yalnizca izinli validation C secer; joint iki dilin Macro-F1 ortalamasini kullanir.',
           'Tamamlanan C-fitler checksum ile tekrar kullanilir; yarim fit yeniden baslar.',
           'PRIVATE modeller, kimlikler, tahminler ve ham metinler paylasim ZIPinde YOK.']
    if 'message' in report:lines.append('Bilgi: '+report['message'])
    atomic_file(out/'linear_training_summary.txt',lambda f:f.write('\n'.join(lines)+'\n'),False)
    def csv_text(rows,columns):
        import csv,io
        buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=columns,extrasaction='ignore');w.writeheader();w.writerows(rows);return buf.getvalue()
    selected=[];trials=[]
    for r in report['completed']:
        for x in r['selections']:
            selected.append({k:x[k] for k in ('seed','context_id','representation','classifier','task','selected_C','selection_score','train_rows','validation_rows','selection_rule')})
        for t in r['trials']:trials.append(dict(t,seed=r['seed'],context_id=r['context_id'],representation=r['representation']))
    sc=['seed','context_id','representation','classifier','task','selected_C','selection_score','train_rows','validation_rows','selection_rule']
    tc=['seed','context_id','representation','classifier','C','validation_score','converged','fit_seconds']
    atomic_file(out/'selected_validation_results.csv',lambda f:f.write(csv_text(selected,sc)),False)
    atomic_file(out/'all_validation_trials.csv',lambda f:f.write(csv_text(trials,tc)),False)
    def zipwrite(f):
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('linear_training_report.json','linear_training_summary.txt','selected_validation_results.csv','all_validation_trials.csv','training_plan.json'):
                if (out/name).exists():z.write(out/name,name)
    atomic_file(out/'shareable_linear_training.zip',zipwrite)
    print('\n'.join(lines),flush=True);print('[SHARE]',out/'shareable_linear_training.zip',flush=True)


def execute(project, split, features, smoke, seeds, representations, families, threads, stop_file=None, hook=None):
    from threadpoolctl import threadpool_limits,threadpool_info
    base,fr,allcontexts=inspect_inputs(split,features,smoke)
    need(set(seeds)<=set(allcontexts),'A requested seed is not in the approved manifests.')
    need(len(seeds)==len(set(seeds)) and len(representations)==len(set(representations)),'Duplicate scope selection.')
    need(set(representations)<=set(REPRESENTATIONS),'Unknown representation scope.')
    tasks=[]
    # Representation-outer order reuses one full frozen bank in RAM across all contexts/seeds.
    for rep in representations:
        for seed in seeds:
            for ctx in allcontexts[seed].values():
                if ctx['context_id'].split(':')[0] in families:tasks.append((rep,seed,ctx))
    need(tasks,'Requested training scope is empty.')
    code=file_sha(__file__)
    spec=dict(base,schema_version=VERSION,runner_sha256=code,C_grid=list(C_GRID),
              lexical_parameters={r:serial_params(vectorizer(r)) for r in ('word','char')},
              classifier_parameters={k:serial_params(classifier(k,1,42)) for k in CLASSIFIERS},
              task='development_only_no_test_scores',no_refit_train_plus_validation=True)
    out=Path(project)/'_revision_linear'/f'linear_{base["protocol_id"]}_{code[:10]}'
    out.mkdir(parents=True,exist_ok=True);print('[OUTPUT]',out,flush=True)
    global_count=sum(len(c) for c in allcontexts.values())*len(REPRESENTATIONS)
    report={'schema_version':VERSION,'created_at':now(),'state':'RUNNING', 'protocol_id':base['protocol_id'],
            'runner_sha256':code,'packages':base['packages'],'requested_representation_jobs':len(tasks),
            'full_scope_representation_jobs':global_count,'requested_seeds':seeds,'requested_representations':representations,
            'requested_families':families,'completed':[],'new_candidate_fits_this_invocation':0,
            'final_test_predictions_created':False,'xlmr_training_performed':False,
            'report_scope':'requested work only; partial scope is never labeled full completion'}
    rc=0
    with lock(out/'.lock'):
        try:
            root_spec=out/'root_spec.json'
            if root_spec.exists():need(load_json(root_spec)==spec,'Run environment/input identity changed. Review before resuming.')
            else:write_json(root_spec,spec)
            write_json(out/'training_plan.json',{'root_spec':spec,'all_contexts':allcontexts,
                 'note':'Only development sets may be loaded; test set IDs/recipe IDs are metadata for later evaluation.'})
            bank=None;alias=None
            def completed_candidate(r):
                report['new_candidate_fits_this_invocation']+=1
                if hook:hook(r)
            with threadpool_limits(limits=threads):
                report['threadpools']=threadpool_info()
                for rep,seed,context in tasks:
                    check_stop(stop_file)
                    folder=out/f'seed_{seed}'/safe_name(context['context_id'])/rep
                    # Matching verified complete task can be reused without re-reading a full bank.
                    train=load_development(split,seed,context['train_set']);val=load_development(split,seed,context['validation_set'])
                    validate_pair(train,val,context)
                    if rep in DIMENSIONS and alias!=rep:
                        bank=None;gc.collect();alias=rep
                        entry=fr['banks'][rep]
                        print('[BANK]',rep,'validating immutable shards; no labels used by bank reader',flush=True)
                        bank=FrozenBank(safe_path(features,entry['bank_directory']),entry,base['corpus_sha256'],base['protocol_id'])
                    elif rep not in DIMENSIONS:bank=None;alias=None
                    print(f'[TRAIN] seed={seed} context={context["context_id"]} rep={rep} n_train={len(train.targets)} n_val={len(val.targets)}',flush=True)
                    result,reused=run_representation(folder,spec,rep,seed,context,train,val,bank,threads,stop_file,completed_candidate)
                    report['completed'].append(result)
                    write_json(out/'linear_training_report.json',report)
                    del train,val;gc.collect()
            report['state']='LINEAR_MODELS_READY' if len(tasks)==global_count else 'REQUESTED_LINEAR_MODELS_READY'
        except BaseException as exc:
            report['state']='INTERRUPTED' if isinstance(exc,(Stopped,KeyboardInterrupt)) else 'LINEAR_TRAINING_BLOCKED'
            report['error_type']=type(exc).__name__
            report['message']=str(exc) if isinstance(exc,TrainingError) else 'Local runtime failure. See PRIVATE log; completed fits retained.'
            atomic_file(out/f'error_{time.time_ns()}.PRIVATE.log',lambda f:f.write(traceback.format_exc()),False)
            rc=130 if report['state']=='INTERRUPTED' else 1
        finally:
            report['finished_at']=now();package_report(out,report)
    return rc,out,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',default='.')
    p.add_argument('--split',required=True)
    p.add_argument('--features',required=True)
    p.add_argument('--smoke',required=True)
    p.add_argument('--seeds',type=int,nargs='+',default=[42,43,44,45,46])
    p.add_argument('--representations',nargs='+',choices=REPRESENTATIONS,default=list(REPRESENTATIONS))
    p.add_argument('--families',nargs='+',choices=('main','binary','low','domain'),default=['main','binary','low','domain'])
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--stop-file',help='Optional sentinel path; remove it explicitly to resume.')
    args=p.parse_args()
    need(args.threads>0,'Positive CPU thread limit required.')
    # Set before importing numerical libraries. No CUDA/torch import or GPU context.
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        os.environ[name]=str(args.threads)
    project=Path(args.project).expanduser().resolve()
    def resolve(x):
        v=Path(x).expanduser();return (v if v.is_absolute() else project/v).resolve()
    signal.signal(signal.SIGTERM,signal_stop);signal.signal(signal.SIGINT,signal_stop)
    rc,_,_=execute(project,resolve(args.split),resolve(args.features),resolve(args.smoke),args.seeds,
                  args.representations,args.families,args.threads,resolve(args.stop_file) if args.stop_file else None)
    return rc

if __name__=='__main__':
    try:sys.exit(main())
    except TrainingError as e:
        print('[BLOCKED]',str(e),file=sys.stderr);sys.exit(1)
