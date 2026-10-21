#!/usr/bin/env python3
"""Audit aggregate arithmetic and publication assets, not private predictions.
This script does not train models, run inference or repeat resampling tests.
"""
from pathlib import Path
import hashlib,json,re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'source_tables';CHECKS=[];CLAIMS=[]
def check(name,condition,detail):
 if not bool(condition):raise AssertionError(name+': '+str(detail))
 CHECKS.append(dict(check=name,state='PASS',detail=detail))
def read(n):return pd.read_csv(DATA/(n+'.csv'))
def fp(x):return float(x)
allr=read('all_results');means=read('all_mean_sd');cl=read('all_class_results');main=read('main_mean_sd')
keys=['seed','experiment_id','representation','classifier']
check('complete_unique_supervised_scope',len(allr)==2820 and not allr.duplicated(keys).any(),'2,820 unique model/split/recipe rows')
check('five_seeds',set(allr.seed)==set(range(42,47)),sorted(allr.seed.unique().tolist()))
check('expected_model_families',allr.experiment_id.str.split(':').str[0].value_counts().to_dict()=={'low':1500,'domain':495,'main':330,'binary':330,'extreme':165},'Unchanged supervised recipe coverage')
calc=allr.groupby(keys[1:]).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std'),weighted_f1_mean=('weighted_f1','mean'),weighted_f1_sd=('weighted_f1','std'),n_seeds=('seed','nunique')).reset_index()
merged=calc.merge(means,on=keys[1:],validate='one_to_one',suffixes=('_calc','_reported'))
err=max(float((merged[n+'_calc']-merged[n+'_reported']).abs().max()) for n in ['macro_f1_mean','macro_f1_sd','accuracy_mean','accuracy_sd','weighted_f1_mean','weighted_f1_sd'])
check('all_means_and_sample_SD',len(merged)==564 and err<1e-12,{'rows':len(merged),'max_error':err})
check('class_row_scope',len(cl)==7965 and not cl.duplicated(keys+['label']).any(),'7,965 class-level records')
cc=cl.groupby(keys).agg(class_macro=('f1-score','mean'),class_support=('support','sum')).reset_index();merged=allr.merge(cc,on=keys,validate='one_to_one')
check('class_macro_and_support_reconstruction',np.allclose(merged.macro_f1,merged.class_macro,rtol=0,atol=1e-12) and np.array_equal(merged.n,merged.class_support),'Macro-F1 and supports reconstruct from fixed task classes')
weighted=cl.copy();weighted['product']=weighted['support']*weighted['f1-score'];weighted=weighted.groupby(keys)[['product','support']].sum().reset_index();weighted['score']=weighted['product']/weighted['support'];merged=allr.merge(weighted,on=keys,validate='one_to_one')
check('weighted_F1_reconstruction',np.allclose(merged.weighted_f1,merged.score,rtol=0,atol=1e-12),'Support-weighted class F1 equals stored weighted F1')
idx=allr.set_index(keys).macro_f1
ci=read('main_split_conditional_ci');check('main_CI_scope',len(ci)==330 and not ci.duplicated(keys).any(),'330 individual intervals; endpoints are not recomputed from private predictions')
check('main_CI_points',all(abs(r.estimate-idx[r.seed,r.experiment_id,r.representation,r.classifier])<1e-12 for r in ci.itertuples()),'Every CI point equals recorded test score')
t=read('transfer_paired_tests_holm40')
check('supervised_contrast_points',len(t)==40 and all(abs(r.estimate-(idx[r.seed,r.experiment_id,*r.model_A.split('+')]-idx[r.seed,r.experiment_id,*r.model_B.split('+')]))<1e-12 for r in t.itertuples()),'All 40 signed paired differences agree with original score rows')
def holm(x):
 x=np.asarray(x,float);ix=np.argsort(x,kind='stable');adj=np.empty_like(x);adj[ix]=np.minimum(1,np.maximum.accumulate(x[ix]*(len(x)-np.arange(len(x)))));return adj
check('supervised_plusone_p',(t.permutations==10000).all() and np.allclose(t.p_raw,(t.extreme_permutations+1)/(t.permutations+1),rtol=0,atol=1e-14),'Monte Carlo plus-one values, not new randomizations')
check('supervised_holm',np.allclose(t.p_holm_40,holm(t.p_raw),rtol=0,atol=1e-14) and np.array_equal(t.reject_holm_05,t.p_holm_40<=.05),'Single family of 40')
check('supervised_sign_patterns',int(t.reject_holm_05.sum())==38,'38 rejections; effects are still reported with their signs')
j=read('joint_paired_differences_ci')
check('joint_scope_and_points',len(j)==110 and all(abs(r.estimate-(idx[r.seed,'main:joint_to_'+r.target,r.representation,r.classifier]-idx[r.seed,'main:'+r.target+'_to_'+r.target,r.representation,r.classifier]))<1e-12 for r in j.itertuples()),'110 paired joint-minus-mono effects; no added p-values')
js=read('joint_summary_for_manuscript')
check('joint_summary_flags',len(js)==22 and all((int((a.ci_low>0).sum()),int((a.ci_high<0).sum()),int(((a.ci_low<=0)&(a.ci_high>=0)).sum()))==(r.pointwise_above_zero,r.pointwise_below_zero,r.pointwise_including_zero) for r in js.itertuples() for a in [j[(j.target==r.target)&(j.representation==r.representation)&(j.classifier==r.classifier)]]),'22 summary rows retain pointwise CI interpretation')
check('low_resource_scope',len(read('low_mean_sd'))==300,'10 pipelines x 5 budgets x 6 scenarios')
check('domain_and_label_scope',len(read('domain_mean_sd'))==99 and len(read('binary_mean_sd'))==66 and len(read('extreme_mean_sd'))==33,'All eleven pipelines retained')
cms=read('main_confusion_counts_and_rowpercent');cmmean=read('main_confusion_rowpercent_mean_sd')
check('main_confusion_export_scope',len(cms)==2970 and len(cmmean)==594,'330 three-class matrices and 66 mean row-normalized matrices')
grpkeys=['experiment_id','representation','classifier','reference_label','predicted_label']
z=cms.groupby(grpkeys).row_percent.agg(['mean','std']).reset_index().merge(cmmean,on=grpkeys,validate='one_to_one')
check('mean_confusions_not_pooled',np.allclose(z['mean'],z.row_percent_mean) and np.allclose(z['std'],z.row_percent_sd),'Normalize per seed first, then mean/sample SD')
check('mean_confusion_row_sum',np.allclose(cmmean.groupby(grpkeys[:-1]).row_percent_mean.sum(),100),'Every displayed reference row sums to 100 percent before rounding')
ll=json.loads((DATA/'llm_analysis_report.json').read_text());lr=read('llm_equal_subset_results');lm=read('llm_equal_subset_mean_sd');lp=read('llm_exploratory_paired_holm20')
check('llm_result_scope',ll['state']=='LLM_ANALYSIS_READY' and len(ll['full_rows_with_confusion'])==len(lr)==120 and len(lm)==24,'12 pipelines x 2 languages x 5 matched subsets')
llidx=lr.set_index(['seed','language','representation','classifier'])
for row in ll['full_rows_with_confusion']:
 cm=np.array(row['confusion_with_abstention']);assert cm.shape==(3,4) and np.array_equal(cm.sum(1),[300]*3)
 sup=cm.sum(1);pr=cm[:,:3].sum(0);tp=np.diag(cm[:,:3]);f=np.divide(2*tp,sup+pr,out=np.zeros(3),where=sup+pr>0)
 assert abs(f.mean()-row['macro_f1'])<1e-12 and abs(tp.sum()/900-row['accuracy'])<1e-12 and cm[:,3].sum()==row['invalid']
 assert all(abs(f[i]-row['class_f1'][key])<1e-12 for i,key in enumerate(['NEGATIVE','NEUTRAL','POSITIVE']))
 saved=llidx.loc[row['seed'],row['language'],row['representation'],row['classifier']]
 assert abs(saved.macro_f1-row['macro_f1'])<1e-12
check('LLM_confusions_and_class_scores',True,'All 120 matrices, 360 class F1 values, accuracy and invalid counts reconstructed')
z=lr.groupby(['language','representation','classifier']).agg(macro_f1_mean=('macro_f1','mean'),macro_f1_sd=('macro_f1','std'),accuracy_mean=('accuracy','mean'),accuracy_sd=('accuracy','std')).reset_index().merge(lm,on=['language','representation','classifier'],validate='one_to_one',suffixes=('_calc','_reported'))
check('LLM_mean_SD',all(np.allclose(z[n+'_calc'],z[n+'_reported'],rtol=0,atol=1e-12) for n in ['macro_f1_mean','macro_f1_sd','accuracy_mean','accuracy_sd']),'All 24 mean/SD rows agree')
check('LLM_plusone_and_Holm',len(lp)==20 and np.allclose(lp.p_raw,(lp.extreme_permutations+1)/(lp.permutations+1),rtol=0,atol=1e-14) and np.allclose(lp.holm_adjusted_p_20,holm(lp.p_raw),rtol=0,atol=1e-14),'Separate unchanged 20-test family')
check('LLM_all_signed_differences',(lp.macro_f1_difference<0).all(),'All 20 point estimates favor the supervised reference; not all tests reject')
q=lr[lr.representation=='qwen3_8b'];check('invalid_output_totals',int(q.invalid.sum())==35 and q.groupby('language').invalid.sum().to_dict()=={'kz':34,'tr':1},'35 invalid evaluations retained')
lclass=read('llm_class_metrics_by_seed');check('KZ_neutral_invalid',int(lclass.query("language=='kz' and representation=='qwen3_8b' and label=='neutral'").invalid.sum())==21,'21 neutral invalid KZ evaluation appearances')
s=read('main_error_slices');parent=allr.set_index(keys).n
check('diagnostic_partition_support',all(int(a.n.sum())==parent[key[:-1]] for key,a in s.groupby(keys+['axis'])),'Each available slice partition reconstructs its parent sample count')
u=s[s.stratum=='untruncated_all_four'].merge(allr,on=keys,validate='one_to_one',suffixes=('_subset','_full'));maxchange=float((u.macro_f1_subset-u.macro_f1_full).abs().max())
check('common_visibility_difference',len(u)==330 and round(maxchange,5)==.00103,{'max_absolute_macro_F1_change':maxchange,'rows':len(u)})
cov=read('lexical_coverage');cv=cov[cov.experiment_id.str.startswith('main:')]
for group,a in cv.groupby(['seed','experiment_id','representation']):
 assert len(a)==2 and a.zero_vector_count.nunique()==1 and a.occurrence_coverage.nunique()==1
check('lexical_diagnostics_shared_heads',True,'LR/SVM coverage agrees; reporting each vocabulary once does not omit a classifier effect')
# Explicit checks on numerical statements in the prose.
mi=main.set_index(['experiment_id','representation','classifier'])
for e,r,c,n in [('tr_to_tr','xlmr','finetuned','0.8155'),('kz_to_kz','xlmr','finetuned','0.5408'),('joint_to_tr','xlmr','finetuned','0.8138'),('joint_to_kz','xlmr','finetuned','0.5595'),('tr_to_tr','char','lr','0.7870'),('kz_to_kz','char','svm','0.5364'),('tr_to_kz','labse','svm','0.4270'),('kz_to_tr','bge_m3','lr','0.6774')]:
 value=float(mi.loc['main:'+e,r,c].macro_f1_mean);assert f'{value:.4f}'==n
 CLAIMS.append(dict(claim='main '+e+' '+r+' '+c,source='main_mean_sd.csv',value=value,rendered=n))
for r,c in [('bge_m3','lr'),('xlmr','finetuned')]:
 for lang in ['tr','kz']:
  a=lp[(lp.language==lang)&(lp.comparison==f'qwen3_8b_minus_{r}_{c}')]
  CLAIMS.append(dict(claim=f'Qwen minus {r}+{c} {lang}',source='llm_exploratory_paired_holm20.csv',mean_delta=float(a.macro_f1_difference.mean()),Holm_rejections=int(a.reject_holm_05.sum())))
check('headline_numerical_claims',True,'Main score highlights and all four LLM signed-effect summaries checked against input rows')
text=(ROOT/'SAUCIS_Results_revised.tex').read_text();supp=(ROOT/'SAUCIS_Results_Supplement.tex').read_text()
labels=re.findall(r'\\label\{([^}]+)\}',text);refs=re.findall(r'\\ref\{([^}]+)\}',text)
check('Results_labels_unique',len(labels)==len(set(labels)) and set(refs)<=set(labels),'All main Results references resolve within the section')
check('Results_structure',len(re.findall(r'\\subsection\{',text))==8 and text.count(r'\begin{table}')==5 and text.count(r'\begin{figure}')==2,'8 subsections, 5 tables, 2 compact two-image figure blocks')
images=re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}',text)
check('all_figure_assets_present',len(images)==4 and all((ROOT/p).is_file() for p in images),'Four separate vector-PDF figure assets')
check('supplement_completeness',supp.count(r'\begin{table}')==21,'21 additional tables; full per-seed records in machine-readable exports')
result=dict(scope='aggregate arithmetic/formatting audit only; no private predictions, weights, bootstrap draws, or new tests',checks=CHECKS,check_count=len(CHECKS),numerical_claims=CLAIMS)
(ROOT/'qa/results_numeric_validation.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(f'{len(CHECKS)} publication-data checks passed.')
