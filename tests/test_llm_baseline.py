"""Tests use fabricated data and dummy tokenizer/model outputs; no real Qwen inference."""
from __future__ import annotations
import importlib.util, sys, hashlib, json, csv, math, os, shutil, sqlite3, tempfile, zipfile
from pathlib import Path
from unittest import mock
import numpy as np
import pytest

HERE=Path(__file__).resolve().parents[1]
sp=importlib.util.spec_from_file_location('llm',HERE/'10_revision_llm_baseline.py')
M=importlib.util.module_from_spec(sp);sys.modules[sp.name]=M;sp.loader.exec_module(M)
SUPPORT=HERE/'08_revision_evaluate_test.py'

@pytest.mark.parametrize('input_,expected',[
 ('NEGATIVE',0),('negative\n',0),(' neutral ',1),('POSITIVE',2),('PoSiTiVe',2),
 ('Olumlu',-1),('negative.',-1),('NEGATIVE because it was bad',-1),
 ('NEGATIVE\nPOSITIVE',-1),('',-1),('POSITIVE:',-1),('UNKNOWN',-1)])
def test_parse_strict(input_,expected):assert M.parse_label(input_)==expected

def test_config_frozen():
 c=M.config();assert c['generation']['do_sample'] is False and c['generation']['enable_thinking'] is False
 assert c['per_class']==300 and len(c['languages'])==2 and len(c['seeds'])==5
 assert c['baseline_is_not_direct_language_to_language_supervised_transfer']

@pytest.mark.parametrize('seed',M.SEEDS)
def test_sample_determinism_balanced(seed):
 labs=np.repeat(np.arange(3),400)
 ids=np.array([hashlib.sha256(f'x{i}'.encode()).hexdigest().encode() for i in range(len(labs))],dtype='S64')
 a=M.pick_indices(seed,'tr',labs,ids)
 b=M.pick_indices(seed,'tr',labs,ids)
 assert a==b and len(a)==900 and len(set(a))==900
 assert [int(sum(labs[a]==i)) for i in range(3)]==[300]*3

@pytest.mark.parametrize('cls',range(3))
def test_sample_no_shrink(cls):
 l=np.repeat(np.arange(3),400);l[l==cls]=(cls+1)%3
 ids=np.array([f'x{i:04d}'.encode() for i in range(len(l))])
 with pytest.raises(M.BaselineError):M.pick_indices(42,'tr',l,ids)

def test_no_prediction_aware_sampling():
 import inspect
 source=inspect.getsource(M.pick_indices)
 assert 'y_pred' not in source and 'macro_f1' not in source and 'score' not in source

def test_metrics_with_abstentions():
 r=M.metrics([0,1,2],[0,-1,2])
 assert r['accuracy']==pytest.approx(2/3)
 assert r['macro_f1']==pytest.approx(2/3)
 assert r['invalid']==1 and r['invalid_rate']==pytest.approx(1/3)
 assert r['confusion_with_abstention']==[[1,0,0,0],[0,0,0,1],[0,0,1,0]]

def test_metrics_no_invalid_matches_sklearn():
 from sklearn.metrics import f1_score,accuracy_score
 rng=np.random.default_rng(912)
 for _ in range(20):
  y=rng.integers(0,3,133);p=rng.integers(0,3,133)
  m=M.metrics(y,p)
  assert m['accuracy']==accuracy_score(y,p)
  assert m['macro_f1']==pytest.approx(f1_score(y,p,labels=[0,1,2],average='macro',zero_division=0))

def test_metrics_invalid_bounds():
 with pytest.raises(M.BaselineError):M.metrics([0,1,2],[0,1,4])

class FakeTokenizer:
 def encode(self,s,add_special_tokens=False):return s.split()
 def decode(self,ids,skip_special_tokens=False,clean_up_tokenization_spaces=False):return ' '.join(ids)
 def apply_chat_template(self,messages,tokenize=False,add_generation_prompt=True,enable_thinking=False):
  assert enable_thinking is False
  return ' '.join(x['content'] for x in messages)+' ASSISTANT:'

def test_prompt_truncation_keeps_instructions():
 t=FakeTokenizer();text=' '.join('verylong' for _ in range(1500))
 p,total,used=M.render_prompt(t,text)
 assert total==1500 and used<=M.REVIEW_TOKEN_CAP and len(t.encode(p))<=M.TOKEN_LIMIT
 assert M.SYSTEM_PROMPT in p and p.endswith('ASSISTANT:')
 assert used==M.REVIEW_TOKEN_CAP

def test_prompt_both_languages_identical_instructions():
 t=FakeTokenizer()
 a,_,_=M.render_prompt(t,'çok güzel ürün')
 b,_,_=M.render_prompt(t,'Өте жақсы өнім')
 assert a.split('Classify the sentiment')[0]==b.split('Classify the sentiment')[0]
 assert M.SYSTEM_PROMPT in a and M.SYSTEM_PROMPT in b

def test_artifact_commit_resume(tmp_path):
 f=tmp_path/'chunk_00000';M.write_jsonl(f/'outputs.PRIVATE.jsonl',[dict(seq=0,predicted=1,raw='NEUTRAL')])
 M.commit(f,'testsig',['outputs.PRIVATE.jsonl'])
 assert M.committed(f,'testsig')
 with pytest.raises(M.BaselineError):M.committed(f,'wrong')
 with (f/'outputs.PRIVATE.jsonl').open('a') as h:h.write('bad')
 with pytest.raises(M.BaselineError):M.committed(f,'testsig')

def test_safe_path_traversal(tmp_path):
 for s in ('../outside','/etc/passwd'):
  with pytest.raises(M.BaselineError):M.safe(tmp_path,s)

def test_pilot_pass_requires_exact_model(tmp_path):
 out=tmp_path;folder=out/'pilot';folder.mkdir()
 M.write(folder/'worker_status.json',dict(state='PASS',total=24,invalid=0))
 model=dict(revision='a'*40,files={'model.safetensors':dict(sha256='b'*64)})
 ident=dict(revision='a'*40,prompt=M.config()['generation'],model_files={'model.safetensors':'b'*64})
 M.write(folder/'identity.json',ident)
 assert M.pilot_status(out,model)['state']=='PASS'
 model['revision']='c'*40
 with pytest.raises(M.BaselineError):M.pilot_status(out,model)

def test_pilot_abstention_block(tmp_path):
 p=tmp_path/'pilot';p.mkdir();M.write(p/'worker_status.json',dict(state='PASS',total=24,invalid=1))
 with pytest.raises(M.BaselineError):M.pilot_status(tmp_path)

def test_download_requires_permission(tmp_path):
 with pytest.raises(M.BaselineError):M.download(tmp_path,False)

@pytest.fixture()
def fake_project(tmp_path):
 project=tmp_path/'project';project.mkdir()
 # Reviewed step08 file is copied unmodified so source SHA gate remains meaningful.
 shutil.copy(SUPPORT,project/M.EVAL_SOURCE[0])
 split=project/'split';test=project/'test';split.mkdir();test.mkdir()
 (split/'split_report.json').write_text('{}')
 db=sqlite3.connect(split/'corpus.local.sqlite3')
 db.execute('CREATE TABLE records(row_id INTEGER PRIMARY KEY,sample_id TEXT,group_id TEXT,model_text TEXT,language TEXT,domain TEXT,label INTEGER,rating INTEGER)')
 # Build 900 test rows per language and seed: exactly 300 of each class.
 data={};records=[];sets={};rid=0
 for seed in M.SEEDS:
  sd=split/f'seed_{seed}';sd.mkdir()
  for lang in M.LANGUAGES:
   rows=[]
   for cls in range(3):
    for i in range(300):
     rid+=1;sample=M.digest(f'{seed}-{lang}-{cls}-{i}');group=M.digest(f'g-{seed}-{lang}-{cls}-{i}');text=f'{lang} synthetic review {cls} example {i}'
     row=(rid,sample,group,text,lang,'NOT_PROVIDED' if lang=='tr' else 'Market',cls,2*cls+1)
     records.append(row);rows.append(row)
   h=hashlib.sha256()
   for r in rows:h.update((r[1]+':'+str(r[6])+'\n').encode())
   key=f'main:{lang}:test';sets[seed,key]=dict(role='test',rows=rows,metadata=dict(manifest_sha256=h.hexdigest(),language_counts={lang:len(rows)}))
 # 24 validation-only reviews for pilot with no label lookup.
 for lang in M.LANGUAGES:
  rows=[]
  for i in range(12):
   rid+=1;rows.append((rid,M.digest(f'val_{lang}_{i}'),M.digest(f'v_{lang}_{i}'),f'{lang} validation text {i}',lang,'NOT_PROVIDED',0,3))
  records+=rows
  sets[42,f'main:{lang}:validation']=dict(role='validation',rows=rows,metadata=dict(manifest_sha256='none',language_counts={lang:len(rows)}))
 db.executemany('INSERT INTO records VALUES(?,?,?,?,?,?,?,?)',records);db.commit();db.close()
 for seed in M.SEEDS:
  sd=split/f'seed_{seed}';conn=sqlite3.connect(sd/'memberships.local.sqlite3')
  conn.execute('CREATE TABLE sets(set_id TEXT PRIMARY KEY,role TEXT,task TEXT,n_rows INTEGER,metadata_json TEXT)')
  conn.execute('CREATE TABLE members(set_id TEXT,row_id INTEGER,target INTEGER)')
  for (s,key),info in sets.items():
   if s!=seed:continue
   conn.execute('INSERT INTO sets VALUES(?,?,?,?,?)',(key,info['role'],'three_class',len(info['rows']),json.dumps(info['metadata'])))
   conn.executemany('INSERT INTO members VALUES(?,?,?)',[(key,r[0],r[6]) for r in info['rows']])
  conn.commit();conn.close()
 plan=[];frozen=dict(state='SELECTIONS_LOCKED_BEFORE_TEST',model_count=1245,plan_sha256=M.digest(M.canonical(plan)))
 M.write(test/'selection_lock.json',frozen);M.write(test/'evaluation_plan.json',plan)
 root=dict(protocol_id='synthetic_protocol',runner_sha256=M.EVAL_SOURCE[1],plan_sha256=frozen['plan_sha256'],
  corpus_sha256=M.filehash(split/'corpus.local.sqlite3'),split_report_sha256=M.filehash(split/'split_report.json'))
 M.write(test/'root_spec.json',root)
 M.write(test/'test_evaluation_report.json',dict(state='FINAL_TEST_RESULTS_READY',protocol_id='synthetic_protocol',
  completed_evaluations=2820,pending_evaluations=[],selection_lock_sha256=M.filehash(test/'selection_lock.json'),
  training_performed=False,model_selection_on_test=False))
 return project,split,test


def test_full_prepare_9000_and_restart(fake_project):
 project,split,test=fake_project;out=project/'output';out.mkdir()
 one=M.prepare(project,split,test,out)
 assert len(list(M.load_jsonl(out/'inference_input.PRIVATE.jsonl')))==9000
 assert len(list(M.load_jsonl(out/'sample_meta.PRIVATE.jsonl')))==9000
 assert all(set(r)=={'seq','text'} for r in M.load_jsonl(out/'inference_input.PRIVATE.jsonl'))
 assert len(list(M.pilot_items(split)))==24
 a=M.validate_prepared(project,split,test,out)
 assert a[-1]['count']==9000
 before=M.filehash(out/'inference_input.PRIVATE.jsonl')
 assert M.prepare(project,split,test,out)==one
 assert M.filehash(out/'inference_input.PRIVATE.jsonl')==before


def test_full_prepare_detects_mutated_source(fake_project):
 project,split,test=fake_project;out=project/'output';out.mkdir();M.prepare(project,split,test,out)
 (split/'split_report.json').write_text('changed')
 with pytest.raises(M.BaselineError):M.validate_prepared(project,split,test,out)


def test_full_prepare_detects_tampered_sample(fake_project):
 project,split,test=fake_project;out=project/'output';out.mkdir();M.prepare(project,split,test,out)
 with (out/'inference_input.PRIVATE.jsonl').open('a') as f:f.write('tamper')
 with pytest.raises(M.BaselineError):M.validate_prepared(project,split,test,out)


def test_cli_worker_optional_stage():
 # Regression: subprocess worker is invoked without an ordinary --stage flag.
 import subprocess
 x=subprocess.run([sys.executable,str(HERE/'10_revision_llm_baseline.py'),'--internal-worker','/nonexistent/x.json'],capture_output=True,text=True)
 assert 'required: --stage' not in x.stderr and x.returncode!=0


def test_shareable_has_no_private_data(tmp_path):
 out=tmp_path;M.write(out/'sampling_summary.json',dict(count=9000))
 (out/'inference_input.PRIVATE.jsonl').write_text('CONFIDENTIAL_TEST_REVIEW')
 M.make_shareable(out,'LLM_SAMPLES_READY')
 with zipfile.ZipFile(out/'shareable_llm_baseline.zip') as z:
  assert all('PRIVATE' not in n for n in z.namelist())
  assert all(b'CONFIDENTIAL_TEST_REVIEW' not in z.read(n) for n in z.namelist())


def test_analysis_120_cells_with_mock_baselines(tmp_path):
 out=tmp_path
 metas=[];outputs=[];baseline=[]
 for seed in M.SEEDS:
  for language in M.LANGUAGES:
   for k in range(900):
    seq=len(metas);lab=k//300
    metas.append(dict(seq=seq,seed=seed,language=language,y_true=lab,group_id=f'{seed}_{language}_{k}'))
    outputs.append(dict(seq=seq,predicted=lab))
    for rep,clf in tuple((r,c) for r in ('word','char','minilm','labse','bge_m3') for c in ('lr','svm'))+(('xlmr','finetuned'),):
     baseline.append(dict(seq=seq,seed=seed,language=language,representation=rep,classifier=clf,predicted=(lab+1)%3))
 M.write_jsonl(out/'sample_meta.PRIVATE.jsonl',metas)
 M.write(out/'evaluation_plan.json',[])
 with mock.patch.object(M,'model_comparison_predictions',return_value=baseline):M.analyze(None,tmp_path,out,outputs)
 d=M.read(out/'llm_analysis_report.json')
 assert d['results']==120 and all(r['n']==900 for r in d['full_rows_with_confusion'])
 assert next(r for r in d['full_rows_with_confusion'] if r['representation']=='qwen3_8b')['macro_f1']==1
 assert len(list(csv.DictReader((out/'llm_equal_subset_mean_sd.csv').open())))==24


def test_paired_same_predictions_null():
 meta=[dict(y_true=i//300,group_id=f'g{i//2}') for i in range(900)]
 preds=[z['y_true'] for z in meta]
 res=M.group_pair_test(42,'tr',meta,preds,preds,'same',bootstrap=103,permutations=311)
 assert res['macro_f1_difference']==res['ci_low']==res['ci_high']==0
 assert res['p_raw']==1 and res['n_text_groups']==450


def test_paired_abstention_and_repeatability():
 meta=[dict(y_true=i//300,group_id=f'group_{i//3}') for i in range(900)]
 a=[-1 if i%9==0 else x['y_true'] for i,x in enumerate(meta)]
 b=[(x['y_true']+1)%3 for x in meta]
 x=M.group_pair_test(45,'kz',meta,a,b,'abstain',bootstrap=150,permutations=501)
 y=M.group_pair_test(45,'kz',meta,a,b,'abstain',bootstrap=150,permutations=501)
 assert x==y and x['n_text_groups']==300
 assert 0<=x['p_raw']<=1 and x['bootstrap_repetitions']==150


def test_holm_20_math_and_permutation():
 v=np.array([.01,.03,.004,.4]);out=M.holm(v)
 assert np.allclose(out,[.03,.06,.016,.4])
 ix=[2,0,3,1];assert np.allclose(out[ix],M.holm(v[ix]))


def test_snapshot_symlink_cache_and_change(tmp_path):
 repo=tmp_path/'models--Qwen--Qwen3-8B';snap=repo/'snapshots'/('a'*40)
 blobs=repo/'blobs';blobs.mkdir(parents=True);snap.mkdir(parents=True)
 (blobs/'hashweight').write_bytes(b'tiny synthetic weights')
 (snap/'model-00001.safetensors').symlink_to('../../blobs/hashweight')
 (snap/'config.json').write_text('{}');(snap/'tokenizer_config.json').write_text('{}')
 payload=dict(repo_id=M.MODEL_ID,revision='a'*40,snapshot=str(snap),files={p.name:M.file_desc(p) for p in M.model_files(snap)})
 M.write(tmp_path/'model_manifest.local.json',payload)
 model,meta=M.get_model(tmp_path)
 assert model==snap
 (blobs/'hashweight').write_bytes(b'tampered')
 with pytest.raises(M.BaselineError):M.get_model(tmp_path)


def test_snapshot_escape_blocked(tmp_path):
 repo=tmp_path/'models--Qwen--Qwen3-8B';snap=repo/'snapshots'/('a'*40)
 snap.mkdir(parents=True)
 forbidden=tmp_path/'external_weights';forbidden.write_bytes(b'fake')
 (snap/'model.safetensors').symlink_to(forbidden)
 (snap/'config.json').write_text('{}');(snap/'tokenizer_config.json').write_text('{}')
 M.write(tmp_path/'model_manifest.local.json',dict(repo_id=M.MODEL_ID,revision='a'*40,snapshot=str(snap),
        files={'model.safetensors':M.file_desc(snap/'model.safetensors')}))
 with pytest.raises(M.BaselineError):M.get_model(tmp_path)


def test_matched_baseline_reference_reader(fake_project):
 project,split,test=fake_project
 out=project/'data';out.mkdir();M.prepare(project,split,test,out)
 meta=list(M.load_jsonl(out/'sample_meta.PRIVATE.jsonl'))
 plan=[];full_by_key={}
 for seed in M.SEEDS:
  for lang in M.LANGUAGES:
   subset=[z for z in meta if z['seed']==seed and z['language']==lang]
   for rep,clf in tuple((r,c) for r in ('word','char','minilm','labse','bge_m3') for c in ('lr','svm'))+(('xlmr','finetuned'),):
    m=dict(model_key=f'{seed}/{lang}/{rep}/{clf}',seed=seed,context_id=f'main:{lang}',representation=rep,classifier=clf,
           model_descriptor=dict(sha256=M.digest(f'{seed}-{lang}-{rep}-{clf}')))
    ev=dict(evaluation_key=m['model_key']+'/x',recipe=dict(experiment_id=f'main:{lang}_to_{lang}'),test_manifest={})
    plan.append(dict(model=m,evaluations=[ev]))
    full_by_key[m['model_key']]=dict(row_ids=np.array([z['row_id'] for z in subset],np.int64),
       sample_ids=np.array([z['sample_id'] for z in subset],dtype='S64'),
       y_true=np.array([z['y_true'] for z in subset],np.int64),y_pred=np.array([z['y_true'] for z in subset],np.int64))
 E=M.load_eval(project)
 def mock_read(folder,signature):
  key=Path(folder).relative_to(test/'evaluations').as_posix().removesuffix('/x')
  return full_by_key[key]
 with mock.patch.object(E,'read_predictions',side_effect=mock_read):
  rows=M.model_comparison_predictions(E,test,plan,meta)
 assert len(rows)==99000 and all(r['predicted']==meta[r['seq']]['y_true'] for r in rows)


def test_cli_prepare_full_fake_source(fake_project):
 project,split,test=fake_project
 import subprocess
 r=subprocess.run([sys.executable,str(HERE/'10_revision_llm_baseline.py'),
 '--stage','prepare','--project',str(project),'--split',str(split),'--test',str(test)],capture_output=True,text=True)
 assert r.returncode==0,r.stderr+'\n'+r.stdout
 assert '[STATE] LLM_SAMPLES_READY' in r.stdout
 outputs=list(project.glob('_revision_llm/llm_*/shareable_llm_baseline.zip'))
 assert len(outputs)==1
 with zipfile.ZipFile(outputs[0]) as z:
  assert 'sampling_summary.json' in z.namelist()
  assert M.read(outputs[0].parent/'sampling_summary.json')['count']==9000
