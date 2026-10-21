#!/usr/bin/env python3
"""SAUCIS step 05 recovery v1.1: convergence-only retry and verified v1 import.

Original outputs are read-only; compatible models are copied byte-for-byte into a
new run root with explicit original producer metadata. Failed SVMs retry from
scratch at 100000 iterations, with all other settings unchanged. Reports include
individual completed candidates in incomplete representation jobs.

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
import copy
import shutil
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

VERSION = '1.1.0'
LEGACY_RUNNER_SHA256 = '6f8b9dbc04ba567e959744ee685134092807a748f769ed1bc586ba1438bb5650'
CONVERGENCE_POLICY = {
    'policy_id': 'svm_convergence_only_10000_then_100000_v1',
    'svm_iteration_caps': [10000, 100000],
    'tol': 1e-4,
    'trigger': 'ConvergenceWarning or iteration cap reached; never validation performance',
    'retry': 'fresh estimator, identical seed, data, C, loss, penalty and dual selection',
    'lr_iteration_caps': [3000],
    'second_failure': 'block; never omit candidate or relax tolerance',
    'legacy_reuse': 'checksum-verified converged fits only; preserve actual producer/parameters',
} 

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


def candidate_identity(spec, kind, C):
    return {'run_spec': spec, 'classifier': kind, 'C': float(C)}


def read_history(folder, identity):
    p = Path(folder)/'attempt_history.json'
    if not p.exists():
        return []
    h = load_json(p)
    need(h['identity_sha256'] == digest(canonical(identity)), 'Attempt history belongs to a different candidate.')
    need(h['records_sha256'] == digest(canonical(h['attempts'])), 'Attempt history checksum mismatch.')
    need(h['policy_id'] == CONVERGENCE_POLICY['policy_id'], 'Attempt policy changed.')
    for a in h['attempts']:
        if a.get('source') == 'legacy_status':
            evidence=Path(folder)/'legacy_failure_evidence.json'
            need(evidence.is_file() and file_sha(evidence)==a['evidence_sha256'],
                 'Legacy failure evidence is missing or changed.')
    return h['attempts']


def save_history(folder, identity, attempts):
    write_json(Path(folder)/'attempt_history.json', {
        'identity_sha256': digest(canonical(identity)),
        'policy_id': CONVERGENCE_POLICY['policy_id'],
        'records_sha256': digest(canonical(attempts)), 'attempts': attempts})


def resolved_dual(model, X):
    if not hasattr(model, 'dual'):
        return None
    # Both primal/dual are supported for the fixed l2+squared_hinge+ovr choice.
    if model.dual == 'auto':
        need(model.penalty == 'l2' and model.loss == 'squared_hinge' and model.multi_class == 'ovr',
             'The audited auto-dual rule applies only to l2/squared_hinge/ovr.')
        return bool(X.shape[0] < X.shape[1])
    return bool(model.dual)


def next_iteration_cap(kind, attempts):
    caps = CONVERGENCE_POLICY['svm_iteration_caps'] if kind == 'svm' else CONVERGENCE_POLICY['lr_iteration_caps']
    failed = {a['max_iter'] for a in attempts if a['state'] == 'NONCONVERGED'}
    if caps[-1] in failed:
        raise TrainingError('Convergence remained blocked at the maximum approved cap. No candidate is omitted.')
    return caps[-1] if caps[0] in failed else caps[0]


def accepted_report(folder, identity, kind, C, seed, context, train=None, val=None):
    """No unpickle: verify completion, numerical metadata and provenance first."""
    folder=Path(folder)
    stamp=verified_bundle(folder, identity)
    if not stamp:
        return None
    required={'classifier.PRIVATE.joblib','validation_predictions.PRIVATE.npz','candidate_report.json'}
    need(required <= set(stamp['outputs']), 'Incomplete candidate completion manifest.')
    r=load_json(folder/'candidate_report.json')
    need(r['identity_sha256']==digest(canonical(identity)) and r['state']=='CANDIDATE_READY'
         and r.get('converged') is True, 'Invalid saved candidate or convergence flag.')
    need(r['C']==float(C) and r['classifier']==kind and r['seed']==seed and r['context_id']==context['context_id'],
         'Saved candidate metadata mismatch.')
    expected=serial_params(classifier(kind,C,seed));actual=r['parameters']
    cap=actual['max_iter'];expected['max_iter']=cap
    need(actual==expected, 'Saved estimator settings differ from approved settings.')
    need(cap in (CONVERGENCE_POLICY['svm_iteration_caps'] if kind=='svm' else [3000]), 'Unexpected iteration cap.')
    values=r['n_iter'] if isinstance(r['n_iter'],list) else [r['n_iter']]
    need(values and all(isinstance(v,(int,float)) and 0 <= v < cap for v in values), 'Unverified numerical termination.')
    need('ConvergenceWarning' not in r.get('warning_categories',[]), 'Saved model has convergence warning.')
    need(r.get('final_test_predictions_created') is False, 'Candidate includes a test evaluation.')
    need(r['validation']['selection_rule']==context['selection_metric']
         and set(r['validation']['by_language'])==set(context['validation_languages'])
         and math.isfinite(r['validation']['selection_score']), 'Invalid permitted validation selection.')
    if train is not None:
        need(r['train_rows']==len(train.targets) and r['validation_rows']==len(val.targets), 'Saved sample counts differ.')
    if r.get('origin') == 'imported_v1':
        need(r['producer_runner_sha256']==LEGACY_RUNNER_SHA256 and cap==(10000 if kind=='svm' else 3000),
             'Invalid original producer or cap.')
        need({'legacy_candidate_report.json','legacy_complete.json','import_provenance.json'} <= set(stamp['outputs']),
             'Imported candidate lacks its original evidence.')
        old=load_json(folder/'legacy_candidate_report.json')
        need(file_sha(folder/'legacy_candidate_report.json')==r['provenance']['source_report_sha256'], 'Original report changed.')
        for k in ('parameters','n_iter','validation','fit_seconds','created_at'):
            need(r[k]==old[k], 'Import altered original scientific result or fit metadata.')
    else:
        need(r.get('producer_runner_sha256')==file_sha(__file__), 'Unknown candidate producer.')
        need('attempt_history.json' in stamp['outputs'], 'Missing completed attempt history.')
        h=read_history(folder,identity)
        need(h and h[-1]['state']=='CONVERGED' and h[-1]['max_iter']==cap, 'Missing successful optimizer attempt.')
    return r


def fit_candidate(folder, spec, kind, C, seed, Xtr, train, Xva, val, context, execution, hook=None):
    import numpy as np
    import joblib
    from sklearn.exceptions import ConvergenceWarning
    folder=Path(folder);identity=candidate_identity(spec,kind,C)
    r=accepted_report(folder,identity,kind,C,seed,context,train,val)
    if r is not None:
        print(f'[REUSE FIT] seed={seed} context={context["context_id"]} rep={spec["representation"]} {kind} C={C:g}',flush=True)
        return r,True
    folder.mkdir(parents=True,exist_ok=True)
    attempts=read_history(folder,identity)
    # A process killed between writes leaves RUNNING, not a successful model.
    for a in attempts:
        if a['state']=='RUNNING':
            a.update(state='INTERRUPTED',note='Previous process ended before candidate commit; fit restarts.')
    save_history(folder,identity,attempts)
    while True:
        try:
            cap=next_iteration_cap(kind,attempts)
        except TrainingError:
            write_json(folder/'status.json',dict(state='CONVERGENCE_BLOCKED',C=float(C),classifier=kind,
                max_iter=attempts[-1]['max_iter'] if attempts else None,identity_sha256=digest(canonical(identity)),
                note='Approved final cap already failed. Review required; no automatic omission.'))
            raise
        model=classifier(kind,C,seed)
        model.set_params(max_iter=cap)
        a=dict(attempt=len(attempts)+1,state='RUNNING',max_iter=cap,tol=model.tol,
               seed=seed,started_at=now(),parameters=serial_params(model),resolved_dual=resolved_dual(model,Xtr),
               producer_runner_sha256=file_sha(__file__),fresh_fit=True,
               trigger='prior_nonconvergence' if cap!= (10000 if kind=='svm' else 3000) else 'initial_or_interrupted_attempt')
        attempts.append(a);save_history(folder,identity,attempts)
        write_json(folder/'status.json',dict(state='FITTING',C=float(C),classifier=kind,max_iter=cap,
                    identity_sha256=digest(canonical(identity)),attempt=a['attempt']))
        action='RETRY' if cap==100000 else 'FIT'
        print(f'[{action}] seed={seed} context={context["context_id"]} rep={spec["representation"]} '
              f'classifier={kind} C={C:g} max_iter={cap} tol={model.tol:g}',flush=True)
        begun=time.monotonic()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(Xtr,train.targets)
        except BaseException as exc:
            a.update(state='INTERRUPTED' if isinstance(exc,(Stopped,KeyboardInterrupt)) else 'ERROR',
                     finished_at=now(),fit_seconds=time.monotonic()-begun,error_type=type(exc).__name__)
            save_history(folder,identity,attempts)
            write_json(folder/'status.json',dict(state=a['state'],C=float(C),classifier=kind,
                       max_iter=cap,identity_sha256=digest(canonical(identity)),error_type=type(exc).__name__))
            raise
        elapsed=time.monotonic()-begun
        nit=np.asarray(model.n_iter_)
        convergence_warning=any(issubclass(w.category,ConvergenceWarning) for w in caught)
        cap_reached=bool(np.max(nit)>=cap)
        a.update(state='NONCONVERGED' if convergence_warning or cap_reached else 'CONVERGED',
                 finished_at=now(),fit_seconds=elapsed,n_iter=nit.tolist(),
                 warning_categories=[w.category.__name__ for w in caught],
                 convergence_warning=convergence_warning,iteration_cap_reached=cap_reached)
        save_history(folder,identity,attempts)
        if a['state']=='CONVERGED':
            break
        write_json(folder/'status.json',dict(state='CONVERGENCE_BLOCKED',C=float(C),classifier=kind,
                   n_iter=a['n_iter'],max_iter=cap,identity_sha256=digest(canonical(identity)),
                   approved_retry_available=(kind=='svm' and cap==10000)))
        if kind=='svm' and cap==10000:
            # No validation prediction has been made; retry trigger is solver-only.
            print('[CONVERGENCE] 10000 cap did not converge; retrying unchanged SVM at 100000.',flush=True)
            continue
        raise TrainingError('A classifier did not converge at the final approved cap. Earlier fits retained; review required.')
    need(np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(),'Nonfinite model parameters.')
    labels=task_labels(train.task);need(model.classes_.tolist()==labels,'Classifier class order mismatch.')
    startpred=time.monotonic();p=model.predict(Xva);decision=np.asarray(model.decision_function(Xva))
    need(np.isfinite(decision).all(),'Nonfinite decision values.')
    metrics=validation_metrics(val,p,context)
    payload={'row_ids':val.row_ids,'sample_ids':val.sample_ids.astype('S64'),'group_ids':val.group_ids.astype('S64'),
             'y_true':val.targets,'y_pred':p.astype(np.int64),'languages':val.languages,
             'domains':val.domains,'decision_scores':decision,'class_ids':model.classes_}
    if kind=='lr':
        proba=model.predict_proba(Xva)
        need(np.isfinite(proba).all() and np.allclose(proba.sum(axis=1),1),'Invalid class probabilities.')
        payload['probabilities']=proba
    pred_seconds=time.monotonic()-startpred
    write_npz(folder/'validation_predictions.PRIVATE.npz',payload)
    write_model(folder/'classifier.PRIVATE.joblib',model)
    reloaded=joblib.load(folder/'classifier.PRIVATE.joblib')
    need(serial_params(reloaded)==serial_params(model),'Model reload changed estimator settings.')
    need(np.array_equal(reloaded.predict(Xva[:min(64,len(val.targets))]),p[:min(64,len(val.targets))]),'Model reload changed predictions.')
    counts=Counter(int(x) for x in train.targets);weights={str(k):len(train.targets)/(len(labels)*counts[k]) for k in labels}
    known_times=[a['fit_seconds'] for a in attempts if a.get('fit_seconds') is not None]
    r={'state':'CANDIDATE_READY','identity_sha256':digest(canonical(identity)),'C':float(C),'classifier':kind,
       'seed':seed,'context_id':context['context_id'],'representation':spec['representation'],
       'task':train.task,'train_rows':len(train.targets),'validation_rows':len(val.targets),
       'train_class_counts':{str(k):v for k,v in sorted(counts.items())},'effective_class_weights':weights,
       'parameters':serial_params(model),'n_iter':np.asarray(model.n_iter_).tolist(),
       'warning_categories':attempts[-1]['warning_categories'],'converged':True,'fit_seconds':elapsed,
       'known_total_optimizer_seconds':sum(known_times),
       'all_optimizer_times_known':all(a.get('fit_seconds') is not None for a in attempts),
       'attempt_count':len(attempts),'convergence_policy':CONVERGENCE_POLICY,
       'validation_prediction_seconds':pred_seconds,'execution':execution,'resolved_dual':resolved_dual(model,Xtr),
       'validation':metrics,'final_test_predictions_created':False,'created_at':now(),
       'origin':'trained_recovery','producer_runner_sha256':file_sha(__file__)}
    write_json(folder/'candidate_report.json',r)
    complete_bundle(folder,identity,['classifier.PRIVATE.joblib','validation_predictions.PRIVATE.npz',
                                    'candidate_report.json','attempt_history.json'] +
                    (['legacy_failure_evidence.json'] if (folder/'legacy_failure_evidence.json').exists() else []))
    write_json(folder/'status.json',dict(state='CANDIDATE_READY',completed_at=now(),classifier=kind,C=float(C),
                                       max_iter=cap,n_iter=r['n_iter'],identity_sha256=digest(canonical(identity))))
    if hook:hook(r)
    return r,False


def representation_spec(base_spec,representation,seed,context,train,val,feature_identity=None):
    return {'root_spec_sha256':digest(canonical(base_spec)),'seed':seed,'context':context,
            'train':train.identity(),'validation':val.identity(),'representation':representation,
            'feature_spec_sha256':feature_identity,'C_grid':list(C_GRID)}


def copy_exact(source,dest):
    source=Path(source);dest=Path(dest)
    if dest.exists():
        need(file_sha(dest)==file_sha(source), 'Import destination differs; no overwrite allowed.')
        return
    before=file_sha(source)
    def writer(f):
        with source.open('rb') as src:shutil.copyfileobj(src,f,4*1024*1024)
    atomic_file(dest,writer)
    need(file_sha(dest)==before==file_sha(source),'Source changed during read-only import.')


def write_once(path,obj):
    if Path(path).exists():
        need(load_json(path)==obj,'Recorded provenance differs; review before resuming.')
    else:write_json(path,obj)


@contextmanager
def read_lock_legacy(root):
    """Shared flock on EXISTING legacy lock; never create or edit legacy files."""
    import fcntl
    p=Path(root)/'.lock'
    need(p.is_file(),'Original lock file missing; legacy output is not recognized.')
    with p.open('r') as f:
        try:fcntl.flock(f,fcntl.LOCK_SH|fcntl.LOCK_NB)
        except BlockingIOError as e:raise TrainingError('Original trainer is running. Stop it before migration.') from e
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)


def legacy_root_spec(base):
    return dict(base,schema_version='1.0.0',runner_sha256=LEGACY_RUNNER_SHA256,C_grid=list(C_GRID),
          lexical_parameters={r:serial_params(vectorizer(r)) for r in ('word','char')},
          classifier_parameters={k:serial_params(classifier(k,1,42)) for k in CLASSIFIERS},
          task='development_only_no_test_scores',no_refit_train_plus_validation=True)


def import_legacy(legacy,out,oldroot,newroot,split,fr,tasks,stop_file=None):
    """Re-envelop checksum-matched results, keeping model bytes and old evidence unchanged.

    Old files are read-only. No fitting or test-label access occurs here. New metadata
    explicitly distinguishes the actual original producer from this importer.
    """
    import numpy as np
    import joblib
    legacy=Path(legacy);out=Path(out)
    need(legacy.resolve()!=out.resolve(),'Recovery must use a separate output directory.')
    need(load_json(legacy/'root_spec.json')==oldroot,'Legacy environment/input/configuration mismatch.')
    provenance_dir=out/'legacy_provenance'
    copy_exact(legacy/'root_spec.json',provenance_dir/'root_spec.json')
    stats=dict(candidate_imported=0,candidate_already_present=0,transformer_imported=0,
               failed_status_imported=0,complete_representation_imported=0)
    for rep,seed,context in tasks:
        check_stop(stop_file)
        rel=Path(f'seed_{seed}')/safe_name(context['context_id'])/rep
        old=legacy/rel;new=out/rel
        if not old.exists():continue
        need((old/'run_spec.json').is_file(),'Legacy run lacks its identity; refuse import.')
        train=load_development(split,seed,context['train_set']);val=load_development(split,seed,context['validation_set'])
        validate_pair(train,val,context)
        fid=fr['banks'][rep]['spec_sha256'] if rep in DIMENSIONS else None
        oldspec=representation_spec(oldroot,rep,seed,context,train,val,fid)
        spec=representation_spec(newroot,rep,seed,context,train,val,fid)
        need(load_json(old/'run_spec.json')==oldspec,'Legacy representation/sample identity differs.')
        write_once(new/'run_spec.json',spec)
        parent_done=verified_bundle(old,oldspec)
        baseprov=dict(source_runner_sha256=LEGACY_RUNNER_SHA256,importer_runner_sha256=file_sha(__file__),
                      source_relative_path=rel.as_posix(),source_root_spec_sha256=digest(canonical(oldroot)),
                      source_run_spec_sha256=file_sha(old/'run_spec.json'),
                      model_bytes_unchanged=True,refit_performed=False)
        if rep in ('word','char'):
            ot=old/'transformer';nt=new/'transformer'
            oi={'run_spec':oldspec,'stage':'train_only_tfidf'};ni={'run_spec':spec,'stage':'train_only_tfidf'}
            stamp=verified_bundle(ot,oi)
            need(stamp is not None,'Missing/corrupt legacy TF-IDF transformer.')
            need({'vectorizer.PRIVATE.joblib','transformer_report.json'} <= set(stamp['outputs']), 'Incomplete legacy transformer.')
            ti=load_json(ot/'transformer_report.json')
            need(ti['train_set']==train.identity() and ti['parameters']==serial_params(vectorizer(rep))
                 and ti['validation_used_for_vocabulary_or_idf'] is False and ti['final_test_accessed'] is False,
                 'Legacy TF-IDF provenance differs.')
            v=joblib.load(ot/'vectorizer.PRIVATE.joblib')
            need(vector_signature(v)==ti['signature'],'Legacy transformer signature mismatch.')
            if not verified_bundle(nt,ni):
                for fn in ('vectorizer.PRIVATE.joblib','transformer_report.json'):copy_exact(ot/fn,nt/fn)
                copy_exact(ot/'complete.json',nt/'legacy_complete.json')
                write_once(nt/'import_provenance.json',dict(baseprov,source_complete_sha256=file_sha(ot/'complete.json')))
                complete_bundle(nt,ni,['vectorizer.PRIVATE.joblib','transformer_report.json','legacy_complete.json','import_provenance.json'])
                stats['transformer_imported']+=1
        for kind in CLASSIFIERS:
            imported_trials=[]
            for i,C in enumerate(C_GRID):
                oc=old/kind/f'C_{i}';nc=new/kind/f'C_{i}'
                oi=candidate_identity(oldspec,kind,C);ni=candidate_identity(spec,kind,C)
                if (oc/'complete.json').exists():
                    stamp=verified_bundle(oc,oi)
                    need({'classifier.PRIVATE.joblib','validation_predictions.PRIVATE.npz','candidate_report.json'} <= set(stamp['outputs']),
                         'Missing original candidate artifacts.')
                    r=load_json(oc/'candidate_report.json')
                    need(r['identity_sha256']==digest(canonical(oi)) and r['state']=='CANDIDATE_READY' and r.get('converged') is True,
                         'Legacy candidate is not successfully converged.')
                    need(r['parameters']==serial_params(classifier(kind,C,seed)) and r['seed']==seed
                         and r['classifier']==kind and r['C']==C and r['context_id']==context['context_id']
                         and r['representation']==rep and r['task']==train.task,
                         'Legacy classifier settings/context do not match.')
                    nit=np.asarray(r['n_iter']);cap=r['parameters']['max_iter']
                    need(nit.size>0 and bool(np.all((nit>=0)&(nit<cap)))
                         and 'ConvergenceWarning' not in r.get('warning_categories',[]),'Legacy convergence not established.')
                    need(r['train_rows']==len(train.targets) and r['validation_rows']==len(val.targets)
                         and r['final_test_predictions_created'] is False,'Legacy rows/test status mismatch.')
                    # Validate fitted object and all saved validation identities/metrics without fitting.
                    model=joblib.load(oc/'classifier.PRIVATE.joblib')
                    need(serial_params(model)==r['parameters'] and np.isfinite(model.coef_).all()
                         and np.isfinite(model.intercept_).all()
                         and np.array_equal(model.classes_,task_labels(train.task))
                         and np.array_equal(np.asarray(model.n_iter_),np.asarray(r['n_iter'])),'Legacy fitted model mismatch.')
                    with np.load(oc/'validation_predictions.PRIVATE.npz',allow_pickle=False) as z:
                        for key,expected in (('row_ids',val.row_ids),('sample_ids',val.sample_ids.astype('S64')),
                                             ('group_ids',val.group_ids.astype('S64')),('y_true',val.targets),
                                             ('languages',val.languages),('domains',val.domains)):
                            need(np.array_equal(z[key],expected),'Legacy validation record identity mismatch.')
                        need(validation_metrics(val,z['y_pred'],context)==r['validation'],'Legacy validation metrics differ.')
                        need(np.isfinite(z['decision_scores']).all(),'Nonfinite legacy decision scores.')
                    existing=accepted_report(nc,ni,kind,C,seed,context,train,val)
                    if existing is None:
                        for fn in ('classifier.PRIVATE.joblib','validation_predictions.PRIVATE.npz'):copy_exact(oc/fn,nc/fn)
                        copy_exact(oc/'candidate_report.json',nc/'legacy_candidate_report.json')
                        copy_exact(oc/'complete.json',nc/'legacy_complete.json')
                        prov=dict(baseprov,source_report_sha256=file_sha(oc/'candidate_report.json'),
                                  source_complete_sha256=file_sha(oc/'complete.json'),
                                  source_model_sha256=file_sha(oc/'classifier.PRIVATE.joblib'))
                        write_once(nc/'import_provenance.json',prov)
                        rr=copy.deepcopy(r)
                        rr.update(identity_sha256=digest(canonical(ni)),origin='imported_v1',
                                  producer_runner_sha256=LEGACY_RUNNER_SHA256,provenance=prov)
                        write_json(nc/'candidate_report.json',rr)
                        complete_bundle(nc,ni,['classifier.PRIVATE.joblib','validation_predictions.PRIVATE.npz',
                             'candidate_report.json','legacy_candidate_report.json','legacy_complete.json','import_provenance.json'])
                        write_json(nc/'status.json',dict(state='CANDIDATE_READY',origin='imported_v1',identity_sha256=digest(canonical(ni))))
                        stats['candidate_imported']+=1
                    else:
                        need(existing.get('origin')=='imported_v1'
                             and existing['provenance']['source_report_sha256']==file_sha(oc/'candidate_report.json'),
                             'A different current candidate conflicts with legacy import.')
                        stats['candidate_already_present']+=1
                    imported_trials.append(load_json(nc/'candidate_report.json'))
                elif (oc/'status.json').is_file():
                    st=load_json(oc/'status.json')
                    if st.get('state')!='CONVERGENCE_BLOCKED':continue
                    need(st.get('classifier')==kind and st.get('C')==C and st.get('max_iter')==(10000 if kind=='svm' else 3000),
                         'Legacy failure does not match candidate settings.')
                    need(np.asarray(st.get('n_iter')).size>0 and bool(np.all(np.asarray(st['n_iter'])>=0)),
                         'Legacy failure has no iteration evidence.')
                    proof=dict(baseprov,source_status_sha256=file_sha(oc/'status.json'),
                               source_status=st,status_only_not_a_saved_model=True)
                    write_once(nc/'legacy_failure_evidence.json',proof)
                    if (nc/'complete.json').exists():continue
                    h=read_history(nc,ni)
                    if not h:
                        h=[dict(attempt=1,state='NONCONVERGED',max_iter=st['max_iter'],n_iter=st['n_iter'],
                                tol=1e-4,seed=seed,parameters=serial_params(classifier(kind,C,seed)),
                                producer_runner_sha256=LEGACY_RUNNER_SHA256,source='legacy_status',
                                evidence_sha256=file_sha(nc/'legacy_failure_evidence.json'),
                                fit_seconds=None,note='Original failed-fit duration was not recorded; not fabricated.')]
                        save_history(nc,ni,h)
                        write_json(nc/'status.json',dict(st,identity_sha256=digest(canonical(ni)),origin='legacy_failure_evidence'))
                        stats['failed_status_imported']+=1
            # A completed classifier may exist inside an incomplete representation.
            sp=old/kind/'selected_config.json'
            if sp.exists() and len(imported_trials)==len(C_GRID):
                selected=load_json(sp);best=winner(imported_trials)
                need(selected['selected_C']==best['C'] and selected['selection_score']==best['validation']['selection_score']
                     and selected['final_test_used_for_selection'] is False
                     and selected['refit_train_plus_validation'] is False,'Legacy selection rule differs.')
                need(selected['selected_model_sha256']==file_sha(safe_path(new,selected['selected_model_relative_path'])),
                     'Legacy selected model checksum differs.')
                write_once(new/kind/'selected_config.json',selected)
        # Preserve a finished parent only after its descendants have all been independently checked/imported.
        if parent_done and not verified_bundle(new,spec):
            with np.load(old/'development_ids.PRIVATE.npz',allow_pickle=False) as z:
                for role,p in (('train',train),('validation',val)):
                    for suffix,x in (('row_ids',p.row_ids),('sample_ids',p.sample_ids.astype('S64')),
                                     ('group_ids',p.group_ids.astype('S64')),('targets',p.targets)):
                        need(np.array_equal(z[f'{role}_{suffix}'],x),'Legacy parent membership payload differs.')
            copy_exact(old/'development_ids.PRIVATE.npz',new/'development_ids.PRIVATE.npz')
            copy_exact(old/'representation_report.json',new/'representation_report.json')
            copy_exact(old/'complete.json',new/'legacy_complete.json')
            write_once(new/'import_provenance.json',dict(baseprov,source_complete_sha256=file_sha(old/'complete.json')))
            for kind in CLASSIFIERS:
                for i,C in enumerate(C_GRID):
                    need(accepted_report(new/kind/f'C_{i}',candidate_identity(spec,kind,C),kind,C,seed,context,train,val) is not None,
                         'Cannot import completed parent with missing candidates.')
            complete_bundle(new,spec,['run_spec.json','development_ids.PRIVATE.npz','representation_report.json',
                                     'lr/selected_config.json','svm/selected_config.json','legacy_complete.json','import_provenance.json'])
            stats['complete_representation_imported']+=1
        del train,val
    write_json(out/'migration_summary.json',dict(state='LEGACY_IMPORT_VERIFIED',source_root_spec_sha256=digest(canonical(oldroot)),
                 source_runner_sha256=LEGACY_RUNNER_SHA256,importer_runner_sha256=file_sha(__file__),**stats))
    print(f'[IMPORT] verified original fits={stats["candidate_imported"]+stats["candidate_already_present"]}; '
          f'newly copied={stats["candidate_imported"]}; failed-status imported={stats["failed_status_imported"]}. '
          'Original directory unchanged.',flush=True)
    return stats


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
                cr=accepted_report(cdir,candidate_identity(spec,kind,C),kind,C,seed,context,train,val)
                need(cr is not None,'Missing verified candidate report.')
                trials.append(cr)
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


def collect_progress(out, report):
    """Count individual committed candidates/selected classifiers, including partial parents."""
    out=Path(out);trials=[];selected=[];attempt_rows=[];complete_jobs=0
    root=load_json(out/'root_spec.json');roothash=digest(canonical(root))
    for job in report.get('requested_tasks',[]):
        seed,rep,context=job['seed'],job['representation'],job['context']
        folder=out/f'seed_{seed}'/safe_name(context['context_id'])/rep
        if not (folder/'run_spec.json').exists():continue
        spec=load_json(folder/'run_spec.json')
        need(spec['root_spec_sha256']==roothash and spec['context']==context and spec['seed']==seed
             and spec['representation']==rep,'Progress scan encountered different run identity.')
        if (folder/'complete.json').exists():
            verified_bundle(folder,spec);complete_jobs+=1
        for kind in CLASSIFIERS:
            successful=[]
            for i,C in enumerate(C_GRID):
                d=folder/kind/f'C_{i}';ident=candidate_identity(spec,kind,C)
                if not d.exists():continue
                r=accepted_report(d,ident,kind,C,seed,context)
                h=read_history(d,ident)
                base=dict(seed=seed,context_id=context['context_id'],representation=rep,classifier=kind,C=C)
                for a in h:
                    attempt_rows.append(dict(base,attempt=a['attempt'],state=a['state'],max_iter=a['max_iter'],
                        tol=a.get('tol'),n_iter=a.get('n_iter'),fit_seconds=a.get('fit_seconds'),
                        source=a.get('source','recovery_runner'),resolved_dual=a.get('resolved_dual'),
                        warning_categories=a.get('warning_categories',[])))
                if r:
                    successful.append(r)
                    trials.append(dict(base,state='CANDIDATE_READY',validation_score=r['validation']['selection_score'],
                        converged=True,fit_seconds=r['fit_seconds'],max_iter=r['parameters']['max_iter'],
                        n_iter=r['n_iter'],origin=r.get('origin'),attempt_count=r.get('attempt_count',1),
                        producer_runner_sha256=r['producer_runner_sha256']))
                    if not h:  # original successful fit: its actual metadata, not a new fit
                        attempt_rows.append(dict(base,attempt=1,state='CONVERGED',max_iter=r['parameters']['max_iter'],
                             tol=r['parameters']['tol'],n_iter=r['n_iter'],fit_seconds=r['fit_seconds'],
                             source='legacy_completed_candidate',resolved_dual=None,
                             warning_categories=r.get('warning_categories',[])))
                else:
                    st=load_json(d/'status.json') if (d/'status.json').is_file() else {'state':'INCOMPLETE'}
                    trials.append(dict(base,state=st['state'],validation_score=None,converged=False,fit_seconds=None,
                        max_iter=st.get('max_iter'),n_iter=st.get('n_iter'),origin='incomplete',attempt_count=len(h)))
            sp=folder/kind/'selected_config.json'
            if sp.exists():
                need(len(successful)==len(C_GRID),'Selection exists with missing/nonconverged C candidates.')
                r=load_json(sp);best=winner(successful)
                need(r['selected_C']==best['C'] and r['selection_score']==best['validation']['selection_score'],
                     'Selection does not equal validation-only winner.')
                need(r['selected_model_sha256']==file_sha(safe_path(folder,r['selected_model_relative_path'])),
                     'Selected model checksum mismatch.')
                selected.append({k:r[k] for k in ('seed','context_id','representation','classifier','task',
                        'selected_C','selection_score','train_rows','validation_rows','selection_rule')})
    return dict(trials=trials,selected=selected,attempts=attempt_rows,complete_jobs=complete_jobs)


def package_report(out, report):
    out=Path(out)
    try:
        progress=collect_progress(out,report)
        report['progress_verified']=True
    except Exception as exc:
        report['state']='LINEAR_TRAINING_BLOCKED';report['progress_verified']=False
        report['report_integrity_error']=type(exc).__name__
        report['message']='Progress/artifact verification failed. Do not use this run until reviewed.'
        atomic_file(out/f'report_error_{time.time_ns()}.PRIVATE.log',lambda f:f.write(traceback.format_exc()),False)
        progress=dict(trials=[],selected=[],attempts=[],complete_jobs=0)
    trials,selected,attempts=progress['trials'],progress['selected'],progress['attempts']
    good=[t for t in trials if t['state']=='CANDIDATE_READY']
    report['completed_C_fits']=len(good)
    report['selected_classifier_count']=len(selected)
    report['complete_representation_count']=progress['complete_jobs']
    report['imported_verified_C_fits']=sum(t['origin']=='imported_v1' for t in good)
    report['candidate_statuses']=trials
    report['optimizer_attempts']=attempts
    report['convergence_policy']=CONVERGENCE_POLICY
    report['nonconverged_attempt_count']=sum(a['state']=='NONCONVERGED' for a in attempts)
    report['pending_or_blocked_candidates']=[t for t in trials if t['state']!='CANDIDATE_READY']
    requested=report['requested_representation_jobs']
    if report['state'] in ('LINEAR_MODELS_READY','REQUESTED_LINEAR_MODELS_READY'):
        if not (len(good)==10*requested and len(selected)==2*requested and progress['complete_jobs']==requested):
            report['state']='LINEAR_TRAINING_BLOCKED';report['message']='Final completion count verification failed.'
    write_json(out/'linear_training_report.json',report)
    lines=['SAUCIS — VALIDATION ILE SECILEN DOGRUSAL MODELLER (YAKINSAMA DUZELTMESI)','='*72,
        'Durum: '+report['state'],'Protocol: '+report['protocol_id'],
        f'Tamamlanan temsil-context isleri: {progress["complete_jobs"]}/{requested}',
        f'Secilmis classifier: {len(selected)}; dogrulanmis tamamlanan C-fit: {len(good)}',
        f'Eski kosudan egitilmeden aktarilan C-fit: {report["imported_verified_C_fits"]}',
        f'Bu cagriya ait yeni basarili C-fit: {report["new_candidate_fits_this_invocation"]}',
        f'Kaydedilmis yakinsamayan deneme: {report["nonconverged_attempt_count"]}',
        'SVM: once 10000; yalnizca yakinsamazsa ayni seed/ayarlarla 100000; tol=1e-4 sabit.',
        'Basarisiz aday sessizce elenmez; LR ayarlari degismedi.',
        'Final-test tahmini/skoru YOK. Orijinal linear klasoru, splitler ve feature bankalari degismedi.',
        'Rapor yarim context icindeki tamamlanan adaylari da sayar.',
        'PRIVATE modeller, ornek kimlikleri ve tahminler paylasim ZIPinde YOK.']
    if 'message' in report:lines.append('Bilgi: '+report['message'])
    atomic_file(out/'linear_training_summary.txt',lambda f:f.write('\n'.join(lines)+'\n'),False)
    def csv_text(rows,columns):
        import csv,io
        buf=io.StringIO(newline='');w=csv.DictWriter(buf,fieldnames=columns,extrasaction='ignore');w.writeheader()
        for r in rows:
            w.writerow({k:canonical(v) if isinstance(v,(list,dict)) else v for k,v in r.items()})
        return buf.getvalue()
    outputs=[('selected_validation_results.csv',selected,['seed','context_id','representation','classifier','task','selected_C',
        'selection_score','train_rows','validation_rows','selection_rule']),
        ('all_validation_trials.csv',trials,['seed','context_id','representation','classifier','C','state','validation_score',
        'converged','fit_seconds','max_iter','n_iter','origin','attempt_count','producer_runner_sha256']),
        ('optimizer_attempts.csv',attempts,['seed','context_id','representation','classifier','C','attempt','state',
        'max_iter','tol','n_iter','fit_seconds','source','resolved_dual','warning_categories'])]
    for name,rows,cols in outputs:atomic_file(out/name,lambda f,rs=rows,cs=cols:f.write(csv_text(rs,cs)),False)
    def zipwrite(f):
        with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
            for name in ('linear_training_report.json','linear_training_summary.txt','selected_validation_results.csv',
                         'all_validation_trials.csv','optimizer_attempts.csv','training_plan.json',
                         'convergence_policy.json','migration_summary.json'):
                if (out/name).is_file():z.write(out/name,name)
    atomic_file(out/'shareable_linear_training.zip',zipwrite)
    print('\n'.join(lines),flush=True);print('[SHARE]',out/'shareable_linear_training.zip',flush=True)


def execute(project,split,features,smoke,seeds,representations,families,threads,stop_file=None,hook=None,
            resume_from=None,prepare_only=False):
    from threadpoolctl import threadpool_limits,threadpool_info
    from contextlib import ExitStack
    base,fr,allcontexts=inspect_inputs(split,features,smoke)
    need(set(seeds)<=set(allcontexts),'A requested seed is not in approved manifests.')
    need(len(seeds)==len(set(seeds)) and len(representations)==len(set(representations)),'Duplicate scope selection.')
    need(set(representations)<=set(REPRESENTATIONS),'Unknown representation scope.')
    tasks=[(rep,seed,ctx) for rep in representations for seed in seeds for ctx in allcontexts[seed].values()
           if ctx['context_id'].split(':')[0] in families]
    need(tasks,'Requested training scope is empty.')
    code=file_sha(__file__);oldspec=legacy_root_spec(base)
    spec=dict(base,schema_version=VERSION,runner_sha256=code,C_grid=list(C_GRID),
              lexical_parameters={r:serial_params(vectorizer(r)) for r in ('word','char')},
              classifier_parameters={k:serial_params(classifier(k,1,42)) for k in CLASSIFIERS},
              task='development_only_no_test_scores',no_refit_train_plus_validation=True,
              convergence_policy=CONVERGENCE_POLICY,
              migration_source_root_sha256=digest(canonical(oldspec)) if resume_from else None)
    out=Path(project)/'_revision_linear'/f'linear_{base["protocol_id"]}_{code[:10]}'
    if resume_from:
        need(Path(resume_from).is_dir(),'Original output folder is missing; no automatic fresh run.')
        need(load_json(Path(resume_from)/'root_spec.json')==oldspec,'Original root specification differs from approved v1 run.')
    out.mkdir(parents=True,exist_ok=True);print('[OUTPUT]',out,flush=True)
    total=sum(len(c) for c in allcontexts.values())*len(REPRESENTATIONS)
    report=dict(schema_version=VERSION,created_at=now(),state='RUNNING',protocol_id=base['protocol_id'],
                runner_sha256=code,packages=base['packages'],requested_representation_jobs=len(tasks),
                full_scope_representation_jobs=total,requested_seeds=seeds,requested_representations=representations,
                requested_families=families,completed=[],new_candidate_fits_this_invocation=0,
                requested_tasks=[dict(representation=r,seed=s,context=c) for r,s,c in tasks],
                final_test_predictions_created=False,xlmr_training_performed=False,
                report_scope='requested work only, including individual fits in incomplete representation jobs')
    rc=0
    with ExitStack() as stack:
        if resume_from:stack.enter_context(read_lock_legacy(resume_from))
        stack.enter_context(lock(out/'.lock'))
        try:
            write_once(out/'root_spec.json',spec)
            write_once(out/'convergence_policy.json',CONVERGENCE_POLICY)
            write_json(out/'training_plan.json',{'root_spec':spec,'all_contexts':allcontexts,
                'note':'Only development set labels may be read. Test IDs are recipe metadata only.'})
            if resume_from:
                report['migration']=import_legacy(resume_from,out,oldspec,spec,split,fr,tasks,stop_file)
            if prepare_only:
                report['state']='RECOVERY_PREPARED';report['message']='Read-only legacy import/verification finished; no new fit.'
            else:
                def completed_candidate(r):
                    report['new_candidate_fits_this_invocation']+=1
                    if hook:hook(r)
                bank=None;alias=None
                with threadpool_limits(limits=threads):
                    report['threadpools']=threadpool_info()
                    for rep,seed,context in tasks:
                        check_stop(stop_file)
                        folder=out/f'seed_{seed}'/safe_name(context['context_id'])/rep
                        train=load_development(split,seed,context['train_set']);val=load_development(split,seed,context['validation_set'])
                        validate_pair(train,val,context)
                        if rep in DIMENSIONS and alias!=rep:
                            bank=None;gc.collect();alias=rep;entry=fr['banks'][rep]
                            print('[BANK]',rep,'validating immutable bank; no labels used by bank reader',flush=True)
                            bank=FrozenBank(safe_path(features,entry['bank_directory']),entry,base['corpus_sha256'],base['protocol_id'])
                        elif rep not in DIMENSIONS:bank=None;alias=None
                        print(f'[TRAIN] seed={seed} context={context["context_id"]} rep={rep} '
                              f'n_train={len(train.targets)} n_val={len(val.targets)}',flush=True)
                        result,reused=run_representation(folder,spec,rep,seed,context,train,val,bank,threads,stop_file,completed_candidate)
                        report['completed'].append(result);write_json(out/'linear_training_report.json',report)
                        del train,val;gc.collect()
                report['state']='LINEAR_MODELS_READY' if len(tasks)==total else 'REQUESTED_LINEAR_MODELS_READY'
        except BaseException as exc:
            report['state']='INTERRUPTED' if isinstance(exc,(Stopped,KeyboardInterrupt)) else 'LINEAR_TRAINING_BLOCKED'
            report['error_type']=type(exc).__name__
            report['message']=str(exc) if isinstance(exc,TrainingError) else 'Local failure. PRIVATE log retained; completed fits preserved.'
            atomic_file(out/f'error_{time.time_ns()}.PRIVATE.log',lambda f:f.write(traceback.format_exc()),False)
            rc=130 if report['state']=='INTERRUPTED' else 1
        finally:
            report['finished_at']=now();package_report(out,report)
            if report['state']=='LINEAR_TRAINING_BLOCKED' and rc==0:rc=1
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
    p.add_argument('--resume-from',required=True,help='Read-only original step-05 output (runner 6f8b9dbc04).')
    p.add_argument('--prepare-only',action='store_true',help='Verify/import saved fits and report progress without training.')
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
                  args.representations,args.families,args.threads,resolve(args.stop_file) if args.stop_file else None,
                  resume_from=resolve(args.resume_from),prepare_only=args.prepare_only)
    return rc

if __name__=='__main__':
    try:sys.exit(main())
    except TrainingError as e:
        print('[BLOCKED]',str(e),file=sys.stderr);sys.exit(1)
