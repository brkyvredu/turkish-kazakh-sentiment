#!/usr/bin/env python3
"""Offline public-artifact audit; Python standard library only.
No training, downloads, model deserialization, inference or statistical resampling.
Existing CIs are checked for coverage/count/point-estimate consistency, not replayed.
"""
from __future__ import annotations
import argparse, ast, csv, gzip, hashlib, json, math, re, statistics, sys
from collections import Counter, defaultdict
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def data(p):
    p=Path(p)
    b=gzip.decompress(p.read_bytes()) if p.suffix=='.gz' else p.read_bytes()
    return json.loads(b)
def rows(p):
    with Path(p).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def near(a,b,tol=1e-11):return abs(float(a)-float(b))<=tol
def check(ok,msg):
    if not ok:raise ValueError(msg)
def metrics(cm):
    k=len(cm);check(k in (2,3),'Fixed reference label count is not 2 or 3.')
    check(all(len(r) in (k,k+1) for r in cm),'Invalid confusion shape.')
    check(all(isinstance(v,int) and v>=0 for r in cm for v in r),'Invalid confusion count.')
    supports=[sum(r) for r in cm];total=sum(supports);check(total>0,'Empty confusion matrix.')
    predicted=[sum(cm[i][j] for i in range(k)) for j in range(k)]
    f=[2*cm[i][i]/(supports[i]+predicted[i]) if supports[i]+predicted[i] else 0.0 for i in range(k)]
    return dict(n=total,accuracy=sum(cm[i][i] for i in range(k))/total,macro_f1=sum(f)/k,
                weighted_f1=sum(v*s for v,s in zip(f,supports))/total,class_f1=f,support=supports)
def holm(ps):
    check(all(math.isfinite(p) and 0<=p<=1 for p in ps),'Invalid p-value.')
    order=sorted(range(len(ps)),key=lambda i:ps[i]);out=[0.0]*len(ps);prev=0.0
    for rank,i in enumerate(order):
        prev=max(prev,(len(ps)-rank)*ps[i]);out[i]=min(prev,1.0)
    return out

def audit(root=ROOT,verify_files=True):
    root=Path(root);checks=[]
    def passed(name,details):checks.append({'check':name,'status':'PASS','details':details})
    # Every original pipeline module is compiled but never executed by this audit.
    fs=data(root/'metadata/frozen_scripts.json')['scripts'];check(len(fs)==12,'Expected 12 frozen experiment scripts.')
    byname={d['path']:d for d in fs}
    for d in fs:
        p=root/d['path'];check(p.is_file() and sha(p)==d['sha256'],'Changed frozen script: '+d['path']);compile(p.read_bytes(),d['path'],'exec')
    passed('frozen_script_bytes_and_syntax',12)
    # Cross-script hard-coded dependency hashes must refer to the actual packaged bytes.
    expected={'05_revision_train_linear.py':'6f8b9dbc04ba567e959744ee685134092807a748f769ed1bc586ba1438bb5650',
       '05_revision_train_linear_recovery.py':'957ffda653a635eb63ea3b3f940f78e350fd4177f90e20f769a2ec10a9603338',
       '06_revision_xlmr_pilot.py':'6c8a833b1c3d5175516f966b6004f2511920576b335df470ed5bba60a970a8da',
       '07_revision_train_xlmr.py':'9e6d39a4a02494f95b09211b13275ad87cf0954b32547ad3b1ae5bde67b2b9e4',
       '08_revision_evaluate_test.py':'6e6a71baef11c3c0ddc1883a8aa0b9d3f25b43ec51a1db25e6a4124b768e23dd',
       '09_revision_analyse_predictions.py':'77ab3023054ef0d3f558a8c44901a3b048ef85597e90718c976a81d3c9dd690c',
       '10_revision_llm_baseline.py':'2b62ff21fe988e1c43834e9b13a6aedb6bf674754a4c3e7ed3ccbc8a32fb7387'}
    check(all(byname[k]['sha256']==v for k,v in expected.items()),'Known pipeline dependency hash differs.')
    passed('known_dependency_hashes',len(expected))
    src=data(root/'provenance/result_sources.json')
    for d in src:
        p=root/d['published_path'];check(sha(p)==d['published_sha256'],'Published evidence checksum differs: '+str(p.relative_to(root)))
        b=gzip.decompress(p.read_bytes()) if d['gzip'] else p.read_bytes()
        check(hashlib.sha256(b).hexdigest()==d['content_sha256'],'Evidence content differs.')
        if not d['redacted_fields']:check(d['content_sha256']==d['original_member_sha256'],'Unmodified evidence differs from original.')
    passed('public_result_provenance',len(src))
    # Original full test aggregate reports are compressed, not replaced by paper-rounded numbers.
    report=data(root/'results/08_test/test_evaluation_report.json.gz')
    check(report['state']=='FINAL_TEST_RESULTS_READY' and not report['pending_evaluations'],'Incomplete test evaluation.')
    rr=report['results'];check(len(rr)==2820,'Unexpected test count.')
    fields=('seed','experiment_id','representation','classifier');key=lambda r:tuple(r[k] for k in fields)
    idx={key(r):r for r in rr};check(len(idx)==2820,'Duplicate test results.')
    check(Counter(r['experiment_id'].split(':')[0] for r in rr)=={'main':330,'binary':330,'extreme':165,'low':1500,'domain':495},'Family coverage differs.')
    passed('test_scope_unique',2820)
    for r in rr:
        m=metrics(r['metrics']['confusion_matrix'])
        check(m['n']==r['metrics']['n'],'Test support differs.')
        for k in ('macro_f1','accuracy','weighted_f1'):check(near(m[k],r['metrics'][k]),'Confusion does not reproduce metric.')
        for i,lab in enumerate(r['metrics']['class_order']):
            cl=r['metrics']['class_report'][lab]
            check(near(m['class_f1'][i],cl['f1-score']) and m['support'][i]==cl['support'],'Class metric/support mismatch.')
        check(not r['trained_or_selected_on_test'] and not r['model_file_modified'],'Recorded test-stage mutation.')
    passed('confusion_and_class_metric_recomputation',2820)
    csv_results=rows(root/'results/08_test/all_test_results.csv')
    for r in csv_results:
        q=idx[int(r['seed']),r['experiment_id'],r['representation'],r['classifier']]
        for k in ('macro_f1','accuracy','weighted_f1'):check(near(r[k],q['metrics'][k]),'CSV/JSON metric differs.')
    check(len(csv_results)==2820,'Test CSV count differs.');passed('test_csv_json_agreement',2820)
    grouped=defaultdict(list)
    for r in rr:grouped[key(r)[1:]].append(r)
    summaries=rows(root/'results/08_test/test_mean_sd.csv');check(len(summaries)==564,'Summary count differs.')
    for s in summaries:
        group=grouped[s['experiment_id'],s['representation'],s['classifier']]
        check(len(group)==5 and {r['seed'] for r in group}==set(range(42,47)),'Five-seed coverage differs.')
        for m in ('macro_f1','accuracy','weighted_f1'):
            x=[r['metrics'][m] for r in group]
            check(near(statistics.mean(x),s[m+'_mean']) and near(statistics.stdev(x),s[m+'_sd']),'Mean/sample SD mismatch.')
    passed('five_partition_means_sample_SDs',564)
    primary=rows(root/'results/09_analysis/tables/transfer_paired_tests_holm40.csv');check(len(primary)==40,'40-test family incomplete.')
    for r in primary:
        a,b=r['model_A'].split('+'),r['model_B'].split('+')
        ma=idx[int(r['seed']),r['experiment_id'],*a]['metrics']['macro_f1'];mb=idx[int(r['seed']),r['experiment_id'],*b]['metrics']['macro_f1']
        check(near(float(r['estimate']),ma-mb),'Transfer point contrast differs.')
        p=(int(r['extreme_permutations'])+1)/(int(r['permutations'])+1)
        check(near(p,r['p_raw']) and int(r['permutations'])==10000,'Monte Carlo plus-one mismatch.')
    adjusted=holm([float(r['p_raw']) for r in primary])
    for r,q in zip(primary,adjusted):check(near(q,r['p_holm_40']) and (q<=.05)==(r['reject_holm_05']=='True'),'Holm40 mismatch.')
    passed('transfer_points_plus_one_Holm40',40)
    single=rows(root/'results/09_analysis/tables/main_split_conditional_ci.csv')
    check(len(single)==330,'Single-CI scope incomplete.')
    for r in single:check(near(r['estimate'],idx[int(r['seed']),r['experiment_id'],r['representation'],r['classifier']]['metrics']['macro_f1']),'CI point differs.')
    passed('single_CI_point_estimates_only',330)
    joint=rows(root/'results/09_analysis/tables/joint_paired_differences_ci.csv');check(len(joint)==110,'Joint scope incomplete.')
    for r in joint:
        s=int(r['seed']);t=r['target'];rep=r['representation'];cl=r['classifier']
        d=idx[s,'main:joint_to_'+t,rep,cl]['metrics']['macro_f1']-idx[s,f'main:{t}_to_{t}',rep,cl]['metrics']['macro_f1']
        check(near(d,r['estimate']),'Joint point differs.')
    check(sum(int(r['seed'])==46 for r in joint)==22,'Last-seed joint rows missing.')
    passed('joint_points_and_final_seed_coverage',110)
    ar=data(root/'results/09_analysis/analysis_report.json')
    check(ar['state']=='POST_TEST_ANALYSIS_READY' and ar['new_units']+ar['reused_units']==810,'Analysis scope differs.')
    check(ar['completed_analysis_units']==788,'Unexpected historical summary-counter value; review its documented correction.')
    passed('documented_nonstatistical_counter_discrepancy','788 historical counter; 810 total units and all 110 joint rows present')
    llm=data(root/'results/10_llm/llm_analysis_report.json');lr=llm['full_rows_with_confusion']
    check(llm['state']=='LLM_ANALYSIS_READY' and len(lr)==120,'LLM result scope incomplete.')
    li={tuple(r[k] for k in ('seed','language','representation','classifier')):r for r in lr}
    check(len(li)==120,'Duplicate LLM-subset results.')
    for r in lr:
        m=metrics(r['confusion_with_abstention'])
        check(m['n']==r['n']==900,'LLM support differs.')
        check(all(v==300 for v in m['support']),'Balanced reference supports differ.')
        for k in ('macro_f1','accuracy'):check(near(m[k],r[k]),'LLM confusion/metric differs.')
        for i,l in enumerate(('NEGATIVE','NEUTRAL','POSITIVE')):check(near(m['class_f1'][i],r['class_f1'][l]),'LLM class F1 differs.')
        check(sum(v[-1] for v in r['confusion_with_abstention'])==r['invalid'],'Invalid count mismatch.')
    check(sum(r['invalid'] for r in lr)==35,'Expected 35 invalid evaluation appearances.')
    passed('LLM_confusions_invalids_class_F1',120)
    lm=rows(root/'results/10_llm/llm_equal_subset_mean_sd.csv');lg=defaultdict(list)
    for r in lr:lg[r['language'],r['representation'],r['classifier']].append(r)
    for r in lm:
        g=lg[r['language'],r['representation'],r['classifier']];check(len(g)==5,'LLM five-seed scope differs.')
        for k in ('macro_f1','accuracy'):
            vals=[z[k] for z in g]
            check(near(statistics.mean(vals),r[k+'_mean']) and near(statistics.stdev(vals),r[k+'_sd']),'LLM mean/SD differs.')
    check(len(lm)==24,'LLM summary scope differs.');passed('LLM_means_sample_SDs',24)
    # Different names are used by the original step10 pairwise CSV.
    pairs=rows(root/'results/10_llm/llm_exploratory_paired_holm20.csv');check(len(pairs)==20,'LLM Holm family incomplete.')
    ps=[]
    for r in pairs:
        ps.append(float(r['p_raw']))
        check(near((int(r['extreme_permutations'])+1)/(int(r['permutations'])+1),r['p_raw']),'LLM Monte Carlo plus-one mismatch.')
        s=int(r['seed']);lang=r['language'];ref={'qwen3_8b_minus_bge_m3_lr':('bge_m3','lr'),'qwen3_8b_minus_xlmr_finetuned':('xlmr','finetuned')}[r['comparison']]
        d=li[s,lang,'qwen3_8b','zero_shot']['macro_f1']-li[s,lang,*ref]['macro_f1']
        check(near(d,r['macro_f1_difference']),'LLM point contrast differs.')
    for r,q in zip(pairs,holm(ps)):check(near(q,r['holm_adjusted_p_20']),'Holm20 mismatch.')
    passed('LLM_points_plus_one_Holm20',20)
    # Public release contents must not include runtime-private data/checkpoints.
    forbidden=[]
    for p in root.rglob('*'):
        if not p.is_file() or any(x in p.parts for x in ('.git','__pycache__','.pytest_cache','_reproduction')):continue
        name=p.name.lower()
        if '.private' in name or name.endswith(('.sqlite3','.sqlite','.npz','.npy','.joblib','.pt','.safetensors','.bin')) or name in ('.env','.netrc','.git-credentials','token','stored_tokens'):
            forbidden.append(str(p.relative_to(root)))
    check(not forbidden,'Private/binary artifacts found: '+', '.join(forbidden));passed('excluded_runtime_artifacts',True)
    if verify_files and (root/'SHA256SUMS.txt').is_file():
        entries=0
        for line in (root/'SHA256SUMS.txt').read_text(encoding='utf-8').splitlines():
            hh,rel=line.split('  ',1);p=root/rel
            check(p.resolve().is_relative_to(root.resolve()) and p.is_file() and sha(p)==hh,'Release file checksum differs: '+rel);entries+=1
        passed('release_file_checksums',entries)
    return {'status':'PASS','checks':checks,'scope':'Public aggregates and frozen-code integrity only; no full GPU retraining, inference or bootstrap/permutation replay.',
        'experiments_reexecuted':False,'conditional_CI_endpoints_independently_recomputed':False,'external_services_contacted':False}

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default=str(ROOT));p.add_argument('--json-out');p.add_argument('--skip-file-manifest',action='store_true')
    a=p.parse_args(argv)
    try:r=audit(Path(a.root).resolve(),not a.skip_file_manifest)
    except (ValueError,KeyError,OSError) as e:print('[FAIL]',str(e),file=sys.stderr);return 1
    if a.json_out:Path(a.json_out).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(r,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
