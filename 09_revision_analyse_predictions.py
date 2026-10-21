#!/usr/bin/env python3
"""SAUCIS step09: read-only analysis of locked step08 predictions.

No models, sklearn fit/predict, torch, network, calibration, or selection. Only
NumPy plus the standard library. Main-task prediction shards are independently
verified; all 2820 aggregate results are checked. CIs are conditional on fixed
trained models and a single split. Overlapping splits are NEVER pooled as 5N.

The four transfer contrasts were listed in the revision planning draft. This
analysis script is finalized after aggregate test results became visible: it is
NOT a preregistration. All contrasts/directions/seeds are retained regardless
of outcome. Secondary diagnostics and joint differences are descriptive.
"""
from __future__ import annotations
import argparse, csv, hashlib, io, json, math, os, signal, sqlite3, sys, time, traceback, unicodedata, zipfile
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

VERSION='1.0.0'
STEP08='6e6a71baef11c3c0ddc1883a8aa0b9d3f25b43ec51a1db25e6a4124b768e23dd'
SEEDS=(42,43,44,45,46)
MODELS=tuple((r,c) for r in ('word','char','minilm','labse','bge_m3') for c in ('lr','svm'))+(('xlmr','finetuned'),)
MAIN=('main:tr_to_tr','main:kz_to_kz','main:tr_to_kz','main:kz_to_tr','main:joint_to_tr','main:joint_to_kz')
CONTRASTS=(('bge_lr_minus_word_lr',('bge_m3','lr'),('word','lr')),
 ('bge_svm_minus_char_svm',('bge_m3','svm'),('char','svm')),
 ('bge_lr_minus_labse_lr',('bge_m3','lr'),('labse','lr')),
 ('xlmr_minus_bge_lr',('xlmr','finetuned'),('bge_m3','lr')))
LIMITS={'minilm':128,'labse':256,'bge_m3':8192,'xlmr':256}
POLICY=dict(primary_metric='macro_f1',bootstrap_repetitions=2000,randomization_repetitions=10000,
 confidence_level=.95,bootstrap_unit='whole_NFKC_casefold_whitespace_text_group',
 bootstrap_method='unstratified_cluster_percentile_fixed_class_macro_F1',
 paired_bootstrap='same_resampled_groups_for_both_models',
 randomization='whole_group_pairwise_swap_two_sided_absolute_delta_plus_one',
 multiplicity='one_Holm_family_all_4_contrasts_x_2_directions_x_5_splits=40',
 alpha=.05,inference_scope='conditional_on_fitted_models_and_one_test_split',
 overlapping_splits_pooled=False,across_seed_pvalue_aggregation=False,
 joint='same_representation_classifier_same_target_manifest_bootstrap_only',
 diagnostics='descriptive_post_test_no_hypothesis_tests',
 length_bands='whitespace_words_1to5_6to15_16to30_31plus',
 script_bands='presence_of_Unicode_named_LATIN_and_CYRILLIC_letters_not_language_ID',
 common_untruncated='intersection_length_within_native_limits_of_all_4_encoders',
 statistical_preregistration_claimed=False,
 contrast_origin='four_pairs_proposed_in_revision_planning;implementation_finalized_after_aggregate_test_results',
 no_refit=True,no_prediction=True,no_model_selection=True)
_STOP=False
class AnalysisError(RuntimeError): pass
class Stopped(AnalysisError): pass
def need(ok,msg):
 if not ok:raise AnalysisError(msg)
def now():
 from datetime import datetime,timezone
 return datetime.now(timezone.utc).isoformat()
def canonical(x):return json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)
def digest(x):return hashlib.sha256(x.encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def safe(root,rel):
 p=Path(rel);need(not p.is_absolute() and '..' not in p.parts,'Unsafe relative path.')
 q=Path(root)/p;need(q.resolve().is_relative_to(Path(root).resolve()),'Path escapes run.');return q
def atomic(p,writer,binary=False):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+f'.partial_{os.getpid()}_{time.time_ns()}')
 with tmp.open('wb' if binary else 'w',**({} if binary else {'encoding':'utf8','newline':''})) as f:
  writer(f);f.flush();os.fsync(f.fileno())
 os.replace(tmp,p)
def write(p,x):atomic(p,lambda f:f.write(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n'))
def csvwrite(p,rows,cols=None):
 rows=list(rows);cols=cols or sorted({k for r in rows for k in r})
 def w(f):
  out=csv.DictWriter(f,fieldnames=cols,extrasaction='raise');out.writeheader();out.writerows(rows)
 atomic(p,w)
def stop(signum,frame):
 global _STOP;_STOP=True
def check_stop():
 if _STOP:raise Stopped('Stopped; finished analysis units are retained.')
@contextmanager
def lock(p,shared=False):
 import fcntl
 p=Path(p)
 if not shared:p.parent.mkdir(parents=True,exist_ok=True)
 need(not shared or p.is_file(),'Source lock file missing.')
 with p.open('r' if shared else 'a') as f:
  try:fcntl.flock(f,(fcntl.LOCK_SH if shared else fcntl.LOCK_EX)|fcntl.LOCK_NB)
  except BlockingIOError as e:raise AnalysisError('Source/output currently has a writer.') from e
  try:yield
  finally:fcntl.flock(f,fcntl.LOCK_UN)
class Verifier:
 def __init__(self):self.cache={}
 def check(self,p,d):
  p=Path(p);need(p.is_file(),'Required committed file missing. No automatic deletion/rebuild.')
  st=p.stat();need(st.st_size==d['bytes'],'Committed file size changed.')
  key=(str(p.resolve()),st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns)
  if key not in self.cache:self.cache[key]=sha(p)
  need(self.cache[key]==d['sha256'],'Committed file checksum changed.')
def bundle(folder,sig,v):
 d=read(Path(folder)/'complete.json');need(d['signature']==sig and d['files'],'Invalid bundle signature.')
 for name,desc in d['files'].items():v.check(safe(folder,name),desc)
 return d

def confusion(y,p,k):
 import numpy as np
 y=np.asarray(y);p=np.asarray(p)
 need(y.ndim==p.ndim==1 and len(y)==len(p) and len(y)>0,'Invalid metric arrays.')
 need(np.issubdtype(y.dtype,np.integer) and np.issubdtype(p.dtype,np.integer),'Labels must be integers.')
 need(np.isin(y,range(k)).all() and np.isin(p,range(k)).all(),'Class out of fixed task schema.')
 return np.bincount(y.astype(np.int64)*k+p.astype(np.int64),minlength=k*k).reshape(k,k)
def scores(cm):
 """Vectorized fixed-class confusion-matrix metrics; last two axes are classes."""
 import numpy as np
 a=np.asarray(cm);need(a.shape[-1]==a.shape[-2] and a.shape[-1] in (2,3),'Invalid confusion shape.')
 need(np.isfinite(a).all() and (a>=0).all(),'Negative/nonfinite confusion counts.')
 t=a.sum((-1,-2));need((t>0).all(),'Empty confusion matrix.')
 sup=a.sum(-1);pr=a.sum(-2);tp=np.diagonal(a,axis1=-2,axis2=-1).astype(float)
 div=lambda x,y:np.divide(x,y,out=np.zeros_like(x,dtype=float),where=y!=0)
 f=div(2*tp,sup+pr)
 return dict(macro_f1=f.mean(-1),accuracy=tp.sum(-1)/t,weighted_f1=(f*sup).sum(-1)/t,
             class_f1=f,precision=div(tp,pr),recall=div(tp,sup),support=sup)
def check_metric(m):
 import numpy as np
 c=np.asarray(m['confusion_matrix']);need(np.issubdtype(c.dtype,np.integer) and c.sum()==m['n'],'Confusion/support mismatch.')
 s=scores(c)
 for name in ('macro_f1','accuracy','weighted_f1'):need(abs(float(s[name])-m[name])<1e-12,'Reported metric does not match confusion matrix.')
 labels=['negative','neutral','positive'] if len(c)==3 else ['negative','positive']
 need(m['class_order']==labels,'Class order changed.')
 for i,label in enumerate(labels):
  q=m['class_report'][label]
  for key,ref in [('support',s['support'][i]),('f1-score',s['class_f1'][i]),('precision',s['precision'][i]),('recall',s['recall'][i])]:need(abs(q[key]-ref)<1e-12,'Class metric mismatch.')
 return c

def audit_report(r,production=True):
 import numpy as np
 need(r['state']=='FINAL_TEST_RESULTS_READY' and not r['pending_evaluations'],'Final evaluation incomplete.')
 need(not r['training_performed'] and not r['model_selection_on_test'],'Test selection/refitting recorded.')
 rows=r['results'];need(len(rows)==r['expected_evaluations']==r['completed_evaluations'],'Evaluation counts differ.')
 if production:need(len(rows)==2820,'Unexpected production scope.')
 keys=set();models={};targets={};grouped=defaultdict(list)
 for x in rows:
  key=(x['seed'],x['experiment_id'],x['representation'],x['classifier']);need(key not in keys,'Duplicate evaluation.');keys.add(key)
  need(x['state']=='TEST_EVALUATION_READY' and not x['trained_or_selected_on_test'] and not x['model_file_modified'],'Unsafe result state.')
  cm=check_metric(x['metrics'])
  for vals in x['strata'].values():
   for m in vals.values():check_metric(m)
   need(np.array_equal(sum(np.array(m['confusion_matrix']) for m in vals.values()),cm),'Strata do not reconstruct overall confusion.')
  mk=(x['seed'],x['context_id'],x['representation'],x['classifier'])
  spec=(x['model_sha256'],x['selected_C'],x['selected_lr'],x['selected_epoch'])
  need(models.setdefault(mk,spec)==spec,'One source context uses different models across targets.')
  tk=(x['seed'],x['test_set']);t=(x['test_manifest_sha256'],x['metrics']['n'],tuple(cm.sum(1)))
  need(targets.setdefault(tk,t)==t,'Models do not share test membership/class support.')
  grouped[key[1:]].append(x)
 if production:
  need(Counter(x['experiment_id'].split(':')[0] for x in rows)=={'main':330,'binary':330,'extreme':165,'low':1500,'domain':495},'Family counts differ.')
  need(len(models)==1245,'Unexpected number of locked model contexts.')
  need(all(len(v)==5 and set(x['seed'] for x in v)==set(SEEDS) for v in grouped.values()),'Missing/duplicate split.')
  expected={(s,e,r,c) for s in SEEDS for e in MAIN for r,c in MODELS}
  need({k for k in keys if k[1].startswith('main:')}==expected,'Main scope incomplete.')
 return dict(state='AGGREGATE_AUDIT_PASS',evaluations=len(rows),five_split_groups=len(grouped),model_contexts=len(models),
             checks='all confusion/class/strata metrics; source model consistency; shared target identity; unique complete scope')

def group_counts(y,pa,groups,k,pb=None):
 import numpy as np
 y=np.asarray(y);pa=np.asarray(pa);g=np.asarray(groups)
 need(len(g)==len(y),'Group alignment error.');confusion(y,pa,k)
 if pb is not None:confusion(y,pb,k)
 ug,inv=np.unique(g,return_inverse=True);n=len(ug);kk=k*k
 a=np.zeros((n,kk),np.int64);np.add.at(a,(inv,y*k+pa),1)
 if pb is None:return a
 b=np.zeros_like(a);np.add.at(b,(inv,y*k+np.asarray(pb)),1)
 return a,b

def compress_groups(a,b=None):
 """Combine identical whole-group confusion signatures, without changing weights.

Multinomial over signature multiplicities equals drawing individual groups with
replacement. Binomial swaps per signature equal independent swaps of its groups.
"""
 import numpy as np
 a=np.asarray(a);need(a.ndim==2 and len(a)>0,'Empty groups.')
 z=a if b is None else np.concatenate((a,b),axis=1)
 unique,count=np.unique(z,axis=0,return_counts=True)
 return unique[:,:a.shape[1]],(None if b is None else unique[:,a.shape[1]:]),count

def stat_seed(identity,kind):return int(digest(canonical([identity,kind]))[:16],16)
def resampling(a,k,identity,b=None,bootstrap=2000,permutations=0):
 import numpy as np
 ua,ub,mult=compress_groups(a,b);ng=int(mult.sum());kk=k*k
 obs_a=a.sum(0).reshape(k,k);obs_b=None if b is None else b.sum(0).reshape(k,k)
 observed=float(scores(obs_a)['macro_f1'])-(0. if b is None else float(scores(obs_b)['macro_f1']))
 rng=np.random.Generator(np.random.PCG64(stat_seed(identity,'bootstrap')));draws=[];missing=0
 for start in range(0,bootstrap,128):
  check_stop();n=min(128,bootstrap-start);counts=rng.multinomial(ng,mult/ng,size=n)
  ca=(counts@ua).reshape(n,k,k);v=scores(ca)['macro_f1']
  if ub is not None:
   cb=(counts@ub).reshape(n,k,k);v=v-scores(cb)['macro_f1']
  missing+=int(np.any(ca.sum(-1)==0,axis=-1).sum());draws.extend(v.tolist())
 need(len(draws)==bootstrap and bootstrap>0,'Bootstrap incomplete.')
 lo,hi=np.quantile(draws,[.025,.975],method='linear')
 out=dict(estimate=observed,ci_low=float(lo),ci_high=float(hi),n_groups=ng,
  bootstrap_repetitions=bootstrap,bootstrap_seed=stat_seed(identity,'bootstrap'),
  missing_class_bootstrap_draws=missing,interval='95% percentile; split/model conditional; not simultaneous',
  aggregate_signatures=len(mult),fixed_labels=list(range(k)))
 if b is not None:
  out.update(macro_f1_A=float(scores(obs_a)['macro_f1']),macro_f1_B=float(scores(obs_b)['macro_f1']),difference='A_minus_B')
 if permutations:
  need(b is not None,'Permutation needs a pair.')
  rng=np.random.Generator(np.random.PCG64(stat_seed(identity,'swap')));extreme=0
  basea=obs_a.ravel();baseb=obs_b.ravel();diff=ub-ua
  for start in range(0,permutations,128):
   check_stop();n=min(128,permutations-start);swaps=rng.binomial(mult,.5,size=(n,len(mult)))
   shift=swaps@diff
   delta=scores((basea+shift).reshape(n,k,k))['macro_f1']-scores((baseb-shift).reshape(n,k,k))['macro_f1']
   extreme+=int((np.abs(delta)>=abs(observed)-1e-12).sum())
  out.update(p_raw=(extreme+1)/(permutations+1),permutations=permutations,extreme_permutations=extreme,
             permutation_seed=stat_seed(identity,'swap'),test='whole-group approximate randomization; two-sided absolute delta')
 return out

def holm(pvalues):
 import numpy as np
 p=np.asarray(pvalues,float);need(p.ndim==1 and len(p)>0 and np.isfinite(p).all() and ((p>=0)&(p<=1)).all(),'Invalid p-values.')
 order=np.argsort(p,kind='stable');q=np.minimum(1,np.maximum.accumulate(p[order]*(len(p)-np.arange(len(p)))))
 out=np.empty_like(q);out[order]=q;return out

def aligned(a,b):
 import numpy as np
 for key in ('row_ids','sample_ids','group_ids','y_true','languages','domains','ratings'):
  need(np.array_equal(a[key],b[key]),'Paired predictions do not have identical examples/labels/groups.')

def read_prediction(test,x,expected_spec,v):
 import numpy as np
 folder=safe(Path(test)/'evaluations',x['evaluation_key']);sig=digest(canonical(expected_spec))
 need(sig==x['signature'],'Evaluation identity differs from locked plan.')
 bundle(folder,sig,v);need(read(folder/'evaluation_spec.json')==expected_spec,'Prediction spec changed.')
 need(read(folder/'evaluation_report.json')==x,'Aggregate/local evaluation report differs.')
 arrays=defaultdict(list);first=0
 required=('row_ids','sample_ids','group_ids','y_true','y_pred','languages','domains','ratings','character_lengths','word_lengths')
 optional=('token_lengths','zero_vector','vocabulary_token_count','analyzed_token_count')
 for i in range(x['shards']):
  check_stop();sf=folder/f'shard_{i:05d}';m=read(sf/'shard_report.json')
  ss=digest(canonical(dict(signature=sig,index=i,lo=m['lo'],hi=m['hi'])))
  need(m['signature']==ss and m['lo']==first and m['hi']-first==m['rows'],'Shard order mismatch.')
  bundle(sf,ss,v)
  with np.load(sf/'predictions.PRIVATE.npz',allow_pickle=False) as z:
   need(set(required)<=set(z.files),'Missing prediction metadata.')
   for key in required+optional:
    if key in z.files:
     values=z[key];need(values.ndim==1 and len(values)==m['rows'],'Invalid prediction column shape.');arrays[key].append(values)
  first=m['hi']
 p={k:np.concatenate(a) for k,a in arrays.items()};n=x['metrics']['n'];need(first==n and all(len(a)==n for a in p.values()),'Incomplete prediction coverage.')
 need(np.all(np.diff(p['row_ids'])>0) and len(np.unique(p['sample_ids']))==n,'Duplicate/unordered prediction identities.')
 h=hashlib.sha256()
 for sid,lab in zip(p['sample_ids'],p['y_true']):
  s=sid.decode() if isinstance(sid,bytes) else str(sid);h.update((s+':'+str(int(lab))+'\n').encode())
 need(h.hexdigest()==x['test_manifest_sha256'],'Prediction membership hash differs.')
 k=3 if x['task']=='three_class' else 2
 need(np.array_equal(confusion(p['y_true'],p['y_pred'],k),np.array(x['metrics']['confusion_matrix'])),'Saved predictions do not reproduce reported confusion.')
 return p

class CorpusMetadata:
 def __init__(self,split,root,v):
  import numpy as np
  self.split=Path(split);corpus=self.split/'corpus.local.sqlite3';need(sha(corpus)==root['corpus_sha256'],'Corpus changed.')
  self.db=sqlite3.connect(corpus.resolve().as_uri()+'?mode=ro',uri=True);self.db.execute('PRAGMA query_only=ON')
  n=self.db.execute('SELECT COUNT(*) FROM records').fetchone()[0]
  self.scripts=np.zeros(n+1,np.uint8);self.length=np.zeros(n+1,np.int32);self.words=np.zeros(n+1,np.int32)
  cache={};seen=0
  for rid,text in self.db.execute('SELECT row_id,model_text FROM records ORDER BY row_id'):
   need(rid==seen+1,'Noncontiguous corpus rows.');seen+=1;flags=0
   for ch in text:
    if ch not in cache:
     name=unicodedata.name(ch,'');cache[ch]=(1 if ch.isalpha() and 'LATIN' in name else 2 if ch.isalpha() and 'CYRILLIC' in name else 0)
    flags|=cache[ch]
   self.scripts[rid]=flags;self.length[rid]=len(text);self.words[rid]=len(text.split())
   if seen%50000==0:check_stop()
  self.members={};self._seed_checked=set()
 def validate(self,x,p):
  import numpy as np
  rid=p['row_ids'];need(np.all((rid>=1)&(rid<len(self.scripts))),'Unknown row ID.')
  need(np.array_equal(p['character_lengths'],self.length[rid]) and np.array_equal(p['word_lengths'],self.words[rid]),'Length metadata differs from corpus.')
  key=(x['seed'],x['test_set'])
  if key not in self.members:
   sd=self.split/f"seed_{x['seed']}";done=read(sd/'complete.json')
   if x['seed'] not in self._seed_checked:
    need(sha(sd/'memberships.local.sqlite3')==done['db_sha256'],'Split membership database changed.');self._seed_checked.add(x['seed'])
   db=sqlite3.connect((sd/'memberships.local.sqlite3').resolve().as_uri()+'?mode=ro',uri=True)
   try:
    m=db.execute('SELECT role,task,n_rows,metadata_json FROM sets WHERE set_id=?',(x['test_set'],)).fetchone()
    need(m and m[0]=='test' and m[1]==x['task'] and m[2]==len(rid),'Incorrect test partition.')
    info=json.loads(m[3]);need(info['manifest_sha256']==x['test_manifest_sha256'],'Test manifest differs.')
    db.execute('ATTACH DATABASE ? AS corpus',((self.split/'corpus.local.sqlite3').resolve().as_uri()+'?mode=ro',))
    rows=db.execute('SELECT r.row_id,r.sample_id,r.group_id,m.target,r.language,r.domain,r.rating FROM members m JOIN corpus.records r USING(row_id) WHERE set_id=? ORDER BY row_id',(x['test_set'],)).fetchall()
   finally:db.close()
   keys=('row_ids','sample_ids','group_ids','y_true','languages','domains','ratings');dtypes=(np.int64,'S64','S64',np.int64,str,str,np.int16)
   self.members[key]={k:np.array([(z[j] if z[j] is not None else -1) for z in rows],dtype=dt) for j,(k,dt) in enumerate(zip(keys,dtypes))}
  for k,a in self.members[key].items():need(np.array_equal(a,p[k]),'Prediction/corpus membership inconsistency.')
 def close(self):self.db.close()

def diagnostic_rows(x,p,script,common):
 import numpy as np
 base={k:x[k] for k in ('seed','experiment_id','representation','classifier','task')};masks=[]
 word=p['word_lengths'];masks.extend(('word_length',label,mask) for label,mask in [('1-5',word<=5),('6-15',(word>=6)&(word<=15)),('16-30',(word>=16)&(word<=30)),('31+ ',word>=31)])
 for flag,label in [(0,'neither_Latin_nor_Cyrillic'),(1,'Latin_no_Cyrillic'),(2,'Cyrillic_no_Latin'),(3,'Latin_and_Cyrillic')]:masks.append(('script',label,script==flag))
 masks.append(('common_encoder_visibility','untruncated_all_four',common));masks.append(('common_encoder_visibility','truncated_at_least_one',~common))
 if 'zero_vector' in p:masks.extend([('lexical_vector','zero',p['zero_vector'].astype(bool)),('lexical_vector','nonzero',~p['zero_vector'].astype(bool))])
 rows=[]
 for axis,value,mask in masks:
  n=int(mask.sum())
  if not n:continue
  cm=confusion(p['y_true'][mask],p['y_pred'][mask],3);s=scores(cm)
  rows.append(base|dict(axis=axis,stratum=value.strip(),n=n,n_groups=int(len(np.unique(p['group_ids'][mask]))),
   macro_f1=float(s['macro_f1']),accuracy=float(s['accuracy']),weighted_f1=float(s['weighted_f1']),
   negative_support=int(s['support'][0]),neutral_support=int(s['support'][1]),positive_support=int(s['support'][2]),
   negative_f1=float(s['class_f1'][0]),neutral_f1=float(s['class_f1'][1]),positive_f1=float(s['class_f1'][2]),
   all_three_labels_present=bool(np.all(s['support']>0)),descriptive_only=True))
 return rows

class Units:
 def __init__(self,out,spec):self.out=Path(out);self.spec=spec;self.v=Verifier();self.new=0;self.reused=0
 def run(self,key,inputs,fn):
  sig=digest(canonical(dict(analysis=self.spec,key=key,inputs=inputs)));folder=self.out/'units'/digest(key)[:24]
  if (folder/'complete.json').exists():
   bundle(folder,sig,self.v);r=read(folder/'result.json');self.reused+=1;return r
  check_stop();r=fn();write(folder/'result.json',r);d=(folder/'result.json').stat()
  write(folder/'complete.json',dict(signature=sig,files={'result.json':dict(bytes=d.st_size,sha256=sha(folder/'result.json'))}))
  self.new+=1;return r

def mean_sd(rows,keys,metrics):
 import numpy as np
 groups=defaultdict(list)
 for r in rows:groups[tuple(r[k] for k in keys)].append(r)
 out=[]
 for key,values in sorted(groups.items()):
  seeds=[r['seed'] for r in values];need(len(seeds)==len(set(seeds)),'Duplicate split in summary.')
  r=dict(zip(keys,key));r.update(n_seeds=len(values),complete_five_seeds=set(seeds)==set(SEEDS))
  for m in metrics:
   a=np.array([r[m] for r in values],float);r[m+'_mean']=float(a.mean());r[m+'_sd']=float(a.std(ddof=1)) if len(a)>1 else None
  out.append(r)
 return out

def descriptive_exports(report,out):
 rows=[];cls=[];diag=[];strata=[]
 for x in report['results']:
  base={k:x[k] for k in ('seed','experiment_id','context_id','representation','classifier','task')}
  rows.append(base|{k:x['metrics'][k] for k in ('n','macro_f1','accuracy','weighted_f1')})
  for lab in x['metrics']['class_order']:cls.append(base|dict(label=lab,**x['metrics']['class_report'][lab]))
  if x['diagnostics']:diag.append(base|x['diagnostics']|dict(n=x['metrics']['n']))
  for ax,vals in x['strata'].items():
   for val,m in vals.items():strata.append(base|dict(axis=ax,stratum=val,**{k:m[k] for k in ('n','macro_f1','accuracy','weighted_f1')}))
 keys=['experiment_id','context_id','representation','classifier','task'];summary=mean_sd(rows,keys,['macro_f1','accuracy','weighted_f1'])
 csvwrite(out/'tables/all_results.csv',rows);csvwrite(out/'tables/all_mean_sd.csv',summary)
 csvwrite(out/'tables/all_class_results.csv',cls);csvwrite(out/'tables/all_strata_results.csv',strata)
 # Diagnostics have different lexical/dense fields; fill absent cells explicitly.
 columns=sorted({k for d in diag for k in d});csvwrite(out/'tables/lexical_coverage.csv',[{k:d.get(k) for k in columns} for d in diag],columns)
 for fam in ('main','low','domain','binary','extreme'):csvwrite(out/f'tables/{fam}_mean_sd.csv',[d for d in summary if d['experiment_id'].startswith(fam+':')])
 table=['% Descriptive mean +/- sample SD over five overlapping grouped splits; not a CI.','\\begin{table*}[t]','\\centering','\\small','\\caption{Three-class test macro-F1 (mean $\\pm$ sample SD over five grouped splits). All classifiers are reported separately.}','\\label{tab:revised_main_results}','\\begin{tabular}{lrrrrrr}','\\hline','Model & TR$\\to$TR & KZ$\\to$KZ & TR$\\to$KZ & KZ$\\to$TR & Joint$\\to$TR & Joint$\\to$KZ \\\\','\\hline']
 idx={(x['experiment_id'],x['representation'],x['classifier']):x for x in summary}
 names={'word':'Word TF-IDF','char':'Char TF-IDF','minilm':'MiniLM','labse':'LaBSE','bge_m3':'BGE-M3','xlmr':'XLM-R'}
 for rep,clf in MODELS:
  vals=[]
  for exp in MAIN:
   d=idx[exp,rep,clf];vals.append(f"${d['macro_f1_mean']:.4f} \\pm {d['macro_f1_sd']:.4f}$")
  name=names[rep]+(' + LR' if clf=='lr' else ' + SVM' if clf=='svm' else ' fine-tuned')
  table.append(name+' & '+' & '.join(vals)+' \\\\')
 table+=['\\hline','\\end{tabular}','\\end{table*}'];atomic(out/'tables/main_results.tex',lambda f:f.write('\n'.join(table)+'\n'))
 return summary

def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project',default='.');p.add_argument('--split',required=True);p.add_argument('--test',required=True);p.add_argument('--threads',type=int,default=4)
 a=p.parse_args(argv);need(a.threads>0,'Positive thread limit required.')
 for var in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[var]=str(a.threads)
 import numpy as np
 project=Path(a.project).expanduser().resolve();resolve=lambda s:(Path(s).expanduser() if Path(s).expanduser().is_absolute() else project/s).resolve()
 test,split=resolve(a.test),resolve(a.split);signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
 with lock(test/'.lock',True):
  report=read(test/'test_evaluation_report.json');root=read(test/'root_spec.json');audit=audit_report(report)
  need(root['runner_sha256']==STEP08 and root['protocol_id']==report['protocol_id'],'Wrong reviewed evaluation producer/protocol.')
  need(np.__version__==root['linear_packages']['numpy'],'NumPy version changed; no automatic upgrade.')
  plan=read(test/'evaluation_plan.json');frozen=read(test/'selection_lock.json');locksha=sha(test/'selection_lock.json')
  need(locksha==report['selection_lock_sha256'] and frozen['state']=='SELECTIONS_LOCKED_BEFORE_TEST' and frozen['model_count']==1245,'Selection lock changed.')
  need(digest(canonical(plan))==root['plan_sha256']==frozen['plan_sha256'],'Evaluation plan changed.')
  need(digest(canonical(frozen['models']))==root['model_registry_sha256'],'Model registry changed.')
  need(sha(split/'split_report.json')==root['split_report_sha256'],'Split report changed.')
  expected={};mkeys=set()
  for j in plan:
   m=j['model'];need(m['model_key'] not in mkeys,'Duplicate locked model.');mkeys.add(m['model_key'])
   for e in j['evaluations']:
    sigspec=dict(selection_lock_sha256=locksha,model_key=m['model_key'],model_sha256=m['model_descriptor']['sha256'],evaluation_key=e['evaluation_key'],recipe=e['recipe'],test_manifest=e['test_manifest'],settings=root['settings'])
    need(e['evaluation_key'] not in expected,'Duplicate evaluation plan key.');expected[e['evaluation_key']]=sigspec
  need(len(expected)==2820 and {x['evaluation_key'] for x in report['results']}==set(expected),'Plan/results differ.')
  for x in report['results']:
   sp=expected[x['evaluation_key']];need(digest(canonical(sp))==x['signature'] and sp['model_sha256']==x['model_sha256'] and sp['recipe']['experiment_id']==x['experiment_id'],'Aggregate result violates locked plan.')
  spec=dict(version=VERSION,runner_sha256=sha(__file__),evaluation_report_sha256=sha(test/'test_evaluation_report.json'),
   selection_lock_sha256=locksha,root_spec_sha256=sha(test/'root_spec.json'),split_report_sha256=root['split_report_sha256'],corpus_sha256=root['corpus_sha256'],
   protocol_id=root['protocol_id'],numpy=np.__version__,unicode=unicodedata.unidata_version,policy=POLICY,contrasts=CONTRASTS)
  out=project/'_revision_analysis'/f"analysis_{root['protocol_id']}_{digest(canonical(spec))[:10]}";print('[OUTPUT]',out,flush=True)
  with lock(out/'.lock'):
   if (out/'analysis_spec.json').exists():need(read(out/'analysis_spec.json')==json.loads(canonical(spec)),'Analysis identity changed.')
   else:write(out/'analysis_spec.json',spec)
   units=Units(out,spec);v=Verifier();meta=None;summary=dict(state='RUNNING',protocol_id=root['protocol_id'],aggregate_audit=audit,main_predictions_verified=0,completed_analysis_units=0,training_or_prediction_performed=False)
   primary=[];single=[];joint=[];diagnostics=[];members_by_seed={};rc=0
   try:
    descriptive_exports(report,out)
    meta=CorpusMetadata(split,root,v)
    lookup={(x['seed'],x['experiment_id'],x['representation'],x['classifier']):x for x in report['results']}
    for seed in SEEDS:
     cache={};members_by_seed[seed]={}
     for exp in MAIN:
      packets={};baseline=None
      for rep,clf in MODELS:
       check_stop();x=lookup[seed,exp,rep,clf];print(f'[VERIFY] seed={seed} {exp} {rep}+{clf}',flush=True)
       data=read_prediction(test,x,expected[x['evaluation_key']],v);meta.validate(x,data)
       if baseline is None:baseline=data
       else:aligned(baseline,data)
       packets[rep,clf]=data;cache[exp,rep,clf]=data;summary['main_predictions_verified']+=1
       g=group_counts(data['y_true'],data['y_pred'],data['group_ids'],3)
       key=f'single:{seed}:{exp}:{rep}:{clf}';inputs=[x['signature'],spec['evaluation_report_sha256']]
       st=units.run(key,inputs,lambda g=g,key=key:resampling(g,3,key));single.append(dict(seed=seed,experiment_id=exp,representation=rep,classifier=clf,**st))
      common=np.ones(len(baseline['y_true']),bool)
      for rep,limit in LIMITS.items():
       clf='finetuned' if rep=='xlmr' else 'lr';length=packets[rep,clf].get('token_lengths');need(length is not None,'Missing token lengths for common visibility.')
       if rep!='xlmr':need(np.array_equal(length,packets[rep,'svm']['token_lengths']),'Token lengths differ between heads.')
       common &= length<=limit
      for rep,clf in MODELS:
       x=lookup[seed,exp,rep,clf];data=packets[rep,clf];key=f'diagnostics:{seed}:{exp}:{rep}:{clf}'
       result=units.run(key,[x['signature']],lambda x=x,data=data:diagnostic_rows(x,data,meta.scripts[data['row_ids']],common))
       diagnostics.extend(result)
      if exp in ('main:tr_to_kz','main:kz_to_tr'):
       for name,ma,mb in CONTRASTS:
        aa=packets[ma];bb=packets[mb];aligned(aa,bb);ga,gb=group_counts(aa['y_true'],aa['y_pred'],aa['group_ids'],3,bb['y_pred'])
        key=f'primary:{seed}:{exp}:{name}';inp=[lookup[seed,exp,*ma]['signature'],lookup[seed,exp,*mb]['signature']]
        st=units.run(key,inp,lambda ga=ga,gb=gb,key=key:resampling(ga,3,key,gb,permutations=10000))
        primary.append(dict(seed=seed,experiment_id=exp,contrast=name,model_A='+'.join(ma),model_B='+'.join(mb),**st))
      if exp in ('main:tr_to_tr','main:kz_to_kz'):members_by_seed[seed][exp]=np.unique(baseline['group_ids'])
      print(f'[MAIN COMPLETE] seed={seed} {exp}',flush=True)
      summary['completed_analysis_units']=units.new+units.reused;write(out/'analysis_report.json',summary)
     for target in ('tr','kz'):
      mono=f'main:{target}_to_{target}';jx=f'main:joint_to_{target}'
      for rep,clf in MODELS:
       aa=cache[jx,rep,clf];bb=cache[mono,rep,clf];aligned(aa,bb);ga,gb=group_counts(aa['y_true'],aa['y_pred'],aa['group_ids'],3,bb['y_pred'])
       key=f'joint:{seed}:{target}:{rep}:{clf}';inp=[lookup[seed,jx,rep,clf]['signature'],lookup[seed,mono,rep,clf]['signature']]
       st=units.run(key,inp,lambda ga=ga,gb=gb,key=key:resampling(ga,3,key,gb))
       joint.append(dict(seed=seed,target=target,representation=rep,classifier=clf,**st))
     cache.clear();meta.members.clear()
    need(len(single)==330 and len(primary)==40 and len(joint)==110 and summary['main_predictions_verified']==330,'Statistical scope incomplete.')
    adjusted=holm([d['p_raw'] for d in primary])
    for d,pv in zip(primary,adjusted):d.update(p_holm_40=float(pv),reject_holm_05=bool(pv<=.05),family_size=40)
    csvwrite(out/'tables/main_split_conditional_ci.csv',single);csvwrite(out/'tables/transfer_paired_tests_holm40.csv',primary);csvwrite(out/'tables/joint_paired_differences_ci.csv',joint)
    csvwrite(out/'tables/main_error_slices.csv',diagnostics)
    csvwrite(out/'tables/main_error_slices_mean_sd.csv',mean_sd(diagnostics,['experiment_id','representation','classifier','task','axis','stratum'],['macro_f1','accuracy','n']))
    csvwrite(out/'tables/joint_delta_mean_sd.csv',mean_sd(joint,['target','representation','classifier'],['estimate']))
    overlaps=[]
    for i,s1 in enumerate(SEEDS):
     for s2 in SEEDS[i+1:]:
      for exp in ('main:tr_to_tr','main:kz_to_kz'):
       a1=members_by_seed[s1][exp];a2=members_by_seed[s2][exp];inter=len(np.intersect1d(a1,a2));union=len(a1)+len(a2)-inter
       overlaps.append(dict(seed_A=s1,seed_B=s2,experiment_id=exp,groups_A=len(a1),groups_B=len(a2),intersection=inter,jaccard=inter/union))
    csvwrite(out/'tables/test_group_overlap_between_splits.csv',overlaps)
    summary.update(state='POST_TEST_ANALYSIS_READY',main_predictions_verified=330,individual_conditional_CIs=330,
      paired_transfer_tests=40,holm_family_size=40,joint_conditional_difference_CIs=110,
      main_diagnostic_rows=len(diagnostics),full_aggregate_evaluations=2820,
      conditional_inference_only=True,preregistration_claimed=False,individual_or_joint_CIs_simultaneous=False,
      unverified_here='No re-inference; non-main results checked at aggregate level; no human label validation.')
   except BaseException as exc:
    rc=130 if isinstance(exc,(Stopped,KeyboardInterrupt)) else 1;summary.update(state='INTERRUPTED' if rc==130 else 'ANALYSIS_BLOCKED',error_type=type(exc).__name__,message=str(exc) if isinstance(exc,AnalysisError) else 'Local runtime failure; inspect PRIVATE traceback locally.')
    atomic(out/f'error_{time.time_ns()}.PRIVATE.log',lambda f:f.write(traceback.format_exc()))
   finally:
    if meta:meta.close()
    summary.update(finished_at=now(),new_units=units.new,reused_units=units.reused);write(out/'analysis_report.json',summary)
    lines=['SAUCIS — KAYITLI TAHMINLERDEN ISTATISTIK / HATA ANALIZI','='*66,'Durum: '+summary['state'],
     'Egitim, GPU, model yukleme, yeni tahmin, model secimi ve veri degisikligi YOK.',
     f"Dogrulanan ana gorev tahmin dosyalari: {summary['main_predictions_verified']}/330",
     f"Yeni analiz birimi: {units.new}; tekrar kullanilan: {units.reused}",
     'CI: sabit egitilmis model ve tek split kosullu. Bes splitin kayitlari bagimsiz havuzlanmadi.',
     'Transfer: 4 sabit kontrast x 2 yon x 5 split; 10000 grup-swap; tek40testHolm.',
     'Bu analiz onkayit/preregistration olarak sunulmaz. Puan/siralama model secimini degistirmez.',
     'Ham yorum, ornek/grup IDleri ve PRIVATE tahminler paylasim ZIPine eklenmez.']
    if summary.get('message'):lines.append(summary['message'])
    atomic(out/'analysis_summary.txt',lambda f:f.write('\n'.join(lines)+'\n'))
    def pack(f):
     with zipfile.ZipFile(f,'w',zipfile.ZIP_DEFLATED) as z:
      for pth in [out/'analysis_spec.json',out/'analysis_report.json',out/'analysis_summary.txt']+sorted((out/'tables').glob('*')):
       if pth.is_file():z.write(pth,str(pth.relative_to(out)))
    atomic(out/'shareable_prediction_analysis.zip',pack,True)
    print('\n'.join(lines),flush=True);print('[SHARE]',out/'shareable_prediction_analysis.zip',flush=True)
   return rc
if __name__=='__main__':
 try:sys.exit(main())
 except AnalysisError as exc:print('[BLOCKED]',str(exc),file=sys.stderr);sys.exit(1)
