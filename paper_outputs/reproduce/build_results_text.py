#!/usr/bin/env python3
"""Rebuild manuscript tables and descriptive prose from supplied aggregate artifacts.
No training, inference, bootstrap, randomization, or model selection is performed.
Run from any working directory. Outputs belong to this publication package only.
"""
from pathlib import Path
import json, math, re
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source_tables'; TABLES=ROOT/'tables'; TABLES.mkdir(exist_ok=True)
read=lambda n:pd.read_csv(DATA/(n+'.csv'))
D={n:read(n) for n in ['main_mean_sd','all_mean_sd','all_results','all_class_results','joint_paired_differences_ci','joint_delta_mean_sd','transfer_paired_tests_holm40','low_mean_sd','domain_mean_sd','binary_mean_sd','extreme_mean_sd','main_error_slices','lexical_coverage','llm_equal_subset_mean_sd','llm_exploratory_paired_holm20','llm_class_metrics_mean_sd','llm_confusion_counts_and_rowpercent','main_confusion_rowpercent_mean_sd']}
MODELS=[(r,c) for r in ['word','char','minilm','labse','bge_m3'] for c in ['lr','svm']]+[('xlmr','finetuned')]
MODELS_LLM=MODELS+[('qwen3_8b','zero_shot')]
NAMES={'word':'Word TF-IDF','char':'Char TF-IDF','minilm':'MiniLM','labse':'LaBSE','bge_m3':'BGE-M3','xlmr':'XLM-R','qwen3_8b':'Qwen3-8B'}
def model(r,c):return NAMES[r]+(' + LR' if c=='lr' else ' + SVM' if c=='svm' else ' fine-tuned' if c=='finetuned' else ' zero-shot')
DIRECTIONS=['tr_to_tr','kz_to_kz','tr_to_kz','kz_to_tr','joint_to_tr','joint_to_kz']
DIRNAMES=[r'TR$\to$TR',r'KZ$\to$KZ',r'TR$\to$KZ',r'KZ$\to$TR',r'Joint$\to$TR',r'Joint$\to$KZ']
def f(x):return f'{float(x):.4f}'
def pm(mean,sd,signed=False):return '$'+(f'{float(mean):+.4f}' if signed else f(mean))+r'\,\pm\,'+f(sd)+'$'
def meanrow(family,exp,r,c):
 d=D[family];q=d[(d.experiment_id==exp)&(d.representation==r)&(d.classifier==c)]
 assert len(q)==1,(family,exp,r,c)
 return q.iloc[0]
def mv(family,exp,r,c):
 x=meanrow(family,exp,r,c);return pm(x.macro_f1_mean,x.macro_f1_sd)
def val(family,exp,r,c):return float(meanrow(family,exp,r,c).macro_f1_mean)
def table(caption,label,headers,rows,align=None,note='',fontsize=r'\small',colsep='4pt'):
 align=align or ('l'+'r'*(len(headers)-1))
 out=[r'\begin{table}[htbp]',r'\centering',fontsize,r'\setlength{\tabcolsep}{'+colsep+'}',r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+align+'}',r'\toprule',' & '.join(headers)+r' \\',r'\midrule']
 for row in rows:out.append(' & '.join(str(a) for a in row)+r' \\')
 out +=[r'\bottomrule',r'\end{tabular}']
 if note:out +=[r'\par\smallskip',r'\begin{minipage}{0.98\linewidth}\footnotesize '+note+r'\end{minipage}']
 out +=[r'\end{table}']
 return '\n'.join(out)+'\n'
def save_table(name,text): (TABLES/(name+'.tex')).write_text(text,encoding='utf8');return text

mainrows=[[model(r,c)]+[mv('main_mean_sd','main:'+e,r,c) for e in DIRECTIONS] for r,c in MODELS]
t_main=save_table('main_results',table('Main three-class test macro-F1: mean $\\pm$ sample SD across the five grouped partitions. All eleven pipelines are reported separately.','tab:main_results',['Model']+DIRNAMES,mainrows,note='The main test sets contain 30,000 Turkish and 36,006 Kazakh records per partition, with the class distributions reported in Methods. Numerical maxima are descriptive; they are not omnibus significance decisions.',fontsize=r'\footnotesize',colsep='2.3pt'))

CONTRASTS=[('bge_lr_minus_word_lr',r'BGE+LR $-$ Word+LR'),('bge_svm_minus_char_svm',r'BGE+SVM $-$ Char+SVM'),('bge_lr_minus_labse_lr',r'BGE+LR $-$ LaBSE+LR'),('xlmr_minus_bge_lr',r'XLM-R $-$ BGE+LR')]
contrast_rows=[]; csummary=[]
for exp,direction in [('main:tr_to_kz',DIRNAMES[2]),('main:kz_to_tr',DIRNAMES[3])]:
 for key,title in CONTRASTS:
  a=D['transfer_paired_tests_holm40'];a=a[(a.experiment_id==exp)&(a.contrast==key)];assert len(a)==5
  pos=int(((a.estimate>0)&a.reject_holm_05).sum());neg=int(((a.estimate<0)&a.reject_holm_05).sum());nr=5-pos-neg
  contrast_rows.append([direction,title,f'{a.estimate.mean():+.4f}',f'[{a.estimate.min():+.4f}, {a.estimate.max():+.4f}]',f'{pos}/{neg}/{nr}'])
  csummary.append(dict(experiment_id=exp,contrast=key,mean_delta=a.estimate.mean(),sd_delta=a.estimate.std(ddof=1),min_delta=a.estimate.min(),max_delta=a.estimate.max(),A_favored=pos,B_favored=neg,not_rejected=nr))
pd.DataFrame(csummary).to_csv(DATA/'transfer_contrast_summary_for_manuscript.csv',index=False)
t_transfer=save_table('transfer_contrasts',table('Partition-specific transfer comparisons. $\\Delta$ is macro-F1 of the first model minus the second.','tab:transfer_contrasts',['Direction','First model $-$ second model',r'Mean $\Delta$',r'Range of $\Delta$','A/B/NR'],contrast_rows,align='llrrr',note='A/B/NR counts partitions favoring the first model, favoring the second model, or not rejecting after the single 40-test Holm correction. The counts are not independent replications. The mean and range summarize effect estimates, not a pooled test or a confidence interval. BGE denotes BGE-M3.',fontsize=r'\footnotesize',colsep='4pt'))

jointrows=[];jsummary=[]
for r,c in MODELS:
 cells=[model(r,c)]
 for target in ['tr','kz']:
  a=D['joint_paired_differences_ci'];a=a[(a.target==target)&(a.representation==r)&(a.classifier==c)];assert len(a)==5
  up=int((a.ci_low>0).sum());down=int((a.ci_high<0).sum());overlap=5-up-down
  cells += [pm(a.estimate.mean(),a.estimate.std(ddof=1),True),f'{up}/{down}/{overlap}']
  jsummary.append(dict(target=target,representation=r,classifier=c,delta_mean=a.estimate.mean(),delta_sd=a.estimate.std(ddof=1),pointwise_above_zero=up,pointwise_below_zero=down,pointwise_including_zero=overlap))
 jointrows.append(cells)
pd.DataFrame(jsummary).to_csv(DATA/'joint_summary_for_manuscript.csv',index=False)
t_joint=save_table('joint_differences',table('Joint minus monolingual macro-F1 on the same target test records. The difference is computed within each partition before averaging.','tab:joint_differences',['Model',r'TR: mean $\Delta\pm$SD','CI +/$-$/0',r'KZ: mean $\Delta\pm$SD','CI +/$-$/0'],jointrows,note=r'CI +/$-$/0 counts the five individual paired 95\% bootstrap intervals that are wholly above zero, wholly below zero, or include zero. These are pointwise conditional intervals, not simultaneous intervals or multiplicity-adjusted hypothesis tests.',fontsize=r'\footnotesize',colsep='4pt'))

DOMS=['TRSAv1','KZ_Market','KZ_Appstore'];domlabel={'TRSAv1':'TRSAv1','KZ_Market':'KZ-Market','KZ_Appstore':'KZ-Appstore'}
domrows=[]
for source in DOMS:
 for target in DOMS:
  exp='domain:'+source+'_to_'+target
  n=int(D['all_results'].query('experiment_id == @exp').n.iloc[0])
  domrows.append([domlabel[source],domlabel[target],f'{n:,}',mv('domain_mean_sd',exp,'bge_m3','lr'),mv('domain_mean_sd',exp,'xlmr','finetuned')])
t_domain=save_table('domain_reference_matrix',table('Domain-restricted results for the two reference pipelines: mean $\\pm$ sample SD macro-F1. Each source uses 1,500 training and 240 validation records.','tab:domain_reference',['Training source','Test target','Test $N$','BGE-M3 + LR','XLM-R'],domrows,align='llrrr',note='These two pipelines are also used as reference comparators in the broader analysis, rather than being selected as the best domain models. The supplementary tables retain all eleven pipelines. Comparisons across different test targets need not share the same distribution.',fontsize=r'\footnotesize',colsep='5pt'))

llmrows=[]
for r,c in MODELS_LLM:
 cells=[model(r,c)]
 for lang in ['tr','kz']:
  a=D['llm_equal_subset_mean_sd'];a=a[(a.language==lang)&(a.representation==r)&(a.classifier==c)];assert len(a)==1
  cells.append(pm(a.iloc[0].macro_f1_mean,a.iloc[0].macro_f1_sd))
 llmrows.append(cells)
t_llm=save_table('llm_balanced_results',table('Matched balanced-subset macro-F1: mean $\\pm$ sample SD over the five evaluation subsets. Each subset has 900 records per language.','tab:llm_balanced_results',['Model','Turkish','Kazakh'],llmrows,note='Each subset contains 300 records per reference class. Supervised models are the monolingual references; Qwen3-8B uses no task-labelled examples. These scores are not directly comparable to scores on the full main test sets, and the comparison does not match supervision budgets.',fontsize=r'\small',colsep='8pt'))

# Main prose uses data-derived values via explicit replacements. No new statistical tests.
text=r'''% Complete replacement for the original Results section only.
% Replace from \section{Results} to immediately before \section{Discussion}.
% Requires graphicx and booktabs. Upload figures/results_revision/ unchanged.
% All values come from the completed revised protocols, not the legacy run.
% No new inferential test, experiment, model, or test-based selection is introduced.

\section{Results}
\label{sec:results}

The supervised evaluation comprised 2,820 model--partition--scenario results. Unless stated otherwise, reported scores are macro-F1 means and sample standard deviations over seeds 42--46. Each pipeline was evaluated using its validation-selected configuration; LR and SVM remain separate rather than being combined into a test-selected representation-family score. Partition-specific intervals and the two hypothesis-test families are interpreted as specified in Methods. The additional LLM results use balanced matched subsets and are reported separately.

\subsection{Overall and Monolingual Performance}
\label{subsec:results_overall}

Table~\ref{tab:main_results} presents the six main scenarios. Fine-tuned XLM-R had the largest mean in both monolingual conditions: 0.8155 for Turkish and 0.5408 for Kazakh. The highest lexical means were 0.7870 for character TF-IDF+LR in Turkish and 0.5364 for character TF-IDF+SVM in Kazakh. The latter was numerically close to XLM-R; no monolingual pairwise test was included in the transfer-test family. Thus, the table identifies numerical rankings without assigning statistical superiority to every maximum.

@@MAIN_TABLE@@

Among the frozen embeddings, BGE-M3+LR and BGE-M3+SVM obtained Turkish monolingual means of 0.7922 and 0.7923, respectively. Their Kazakh means were 0.5150 and 0.5337. Classifier choice was therefore not interchangeable: the Kazakh LR and SVM results differed within the same representation, while the two Turkish BGE-M3 means were close. XLM-R also had the largest joint-training means, 0.8138 for Turkish and 0.5595 for Kazakh. The effects of adding the other language are examined separately below.

\subsection{Cross-Lingual Transfer and Paired Comparisons}
\label{subsec:results_transfer}

Direct transfer produced a different ordering from monolingual evaluation. In TR$\rightarrow$KZ, lexical macro-F1 means ranged from 0.0997 to 0.1104, compared with 0.3230--0.4270 for the frozen multilingual pipelines. LaBSE+SVM had the largest numerical mean, 0.4270, followed by BGE-M3+SVM at 0.4195; XLM-R obtained 0.4129. In KZ$\rightarrow$TR, BGE-M3+LR had the largest mean, 0.6774, compared with 0.6305 for XLM-R and a maximum lexical mean of 0.3202 for character TF-IDF+LR. These directions evaluate different target corpora and should not be interpreted as a controlled measure of language difficulty.

The four specified contrasts are summarized in Table~\ref{tab:transfer_contrasts}. BGE-M3+LR exceeded word TF-IDF+LR, and BGE-M3+SVM exceeded character TF-IDF+SVM, in both directions in every partition. All twenty corresponding tests rejected after the 40-test Holm correction (adjusted $p\approx0.0040$). The mean differences were 0.3164 and 0.3091 in TR$\rightarrow$KZ, and 0.4668 and 0.4071 in KZ$\rightarrow$TR.

@@TRANSFER_TABLE@@

The multilingual-model comparisons were direction-dependent. In KZ$\rightarrow$TR, BGE-M3+LR exceeded LaBSE+LR by 0.0505 and XLM-R by 0.0470 on average, with the same direction and Holm-adjusted rejection in all five partitions. In TR$\rightarrow$KZ, however, BGE-M3+LR versus LaBSE+LR favored BGE-M3 in two partitions and LaBSE in one after correction; the other two tests did not reject. Their adjusted $p$-values were approximately 0.0578. XLM-R versus BGE-M3+LR rejected in every TR$\rightarrow$KZ partition, but two differences favored XLM-R and three favored BGE-M3. A significant difference within each partition therefore did not establish a stable cross-partition ordering. No equivalence conclusion follows from a non-rejected test.

\subsection{Effects of Joint Training}
\label{subsec:results_joint}

Table~\ref{tab:joint_differences} reports joint-minus-monolingual differences for every pipeline on the same target records. For Kazakh, XLM-R increased from 0.5408 to 0.5595, a mean difference of +0.0187; its five individual paired intervals were above zero. In contrast, BGE-M3+SVM decreased from 0.5337 to 0.5241, with a mean difference of $-0.0095$ and five intervals below zero. BGE-M3+LR had a smaller mean increase of +0.0021, with only one interval wholly above zero. Word TF-IDF+LR also showed a positive Kazakh difference, +0.0124, with five intervals above zero.

@@JOINT_TABLE@@

For Turkish, joint training reduced the mean score of each frozen-embedding pipeline. XLM-R changed by $-0.0017$, with four of its five paired intervals including zero. Character TF-IDF+SVM changed by +0.0005, also with four intervals including zero. These model-specific patterns do not support a uniform improvement from joint training. The interval counts in Table~\ref{tab:joint_differences} are descriptive summaries of pointwise conditional intervals, not multiplicity-adjusted significance decisions.

\subsection{Low-Resource Evaluation}
\label{subsec:results_low_resource}

The expanded low-resource evaluation included all ten lexical/frozen-embedding pipelines at every label budget. Figure~\ref{fig:low_resource_lr} shows the five representations with the same LR classifier in the monolingual scenarios; complete LR/SVM results for all six scenarios and budgets are included in the supplementary tables. The budget $n$ counts training plus validation records per class and language, not training records alone.

\begin{figure}[htbp]
\centering
\begin{minipage}{0.49\linewidth}\centering
\includegraphics[width=\linewidth]{figures/results_revision/low_tr_to_tr_lr.pdf}
\end{minipage}\hfill
\begin{minipage}{0.49\linewidth}\centering
\includegraphics[width=\linewidth]{figures/results_revision/low_kz_to_kz_lr.pdf}
\end{minipage}
\caption{Monolingual low-resource results with LR fixed across representations. Points are five-partition means; error bars are sample SD, not confidence intervals. The horizontal axis is logarithmic and shows the total training-plus-validation budget per class and language. SVM results are retained in the supplementary tables rather than selected or omitted according to test scores.}
\label{fig:low_resource_lr}
\end{figure}

At $n=100$, BGE-M3+LR obtained 0.7248 in Turkish and 0.4963 in Kazakh, compared with 0.6187 and 0.3989 for word TF-IDF+LR. At $n=5000$, the corresponding means were 0.7863 and 0.5124 for BGE-M3+LR, and 0.7643 and 0.4853 for word TF-IDF+LR. The Turkish scores increased over these endpoints for both pipelines. The Kazakh BGE-M3+LR curve was not monotonic across intermediate budgets, so additional labels did not produce a uniform increment at every step.

Transfer remained substantially weaker for the lexical references. From $n=100$ to $n=5000$, character TF-IDF+SVM changed from 0.0869 to 0.1079 in TR$\rightarrow$KZ and from 0.2390 to 0.3712 in KZ$\rightarrow$TR. BGE-M3+LR obtained 0.4200 and 0.4172 at those TR$\rightarrow$KZ endpoints, and 0.6332 and 0.6680 in KZ$\rightarrow$TR. These are descriptive learning-curve results, not additional paired hypothesis tests. Joint low-resource conditions use twice the total annotation budget of the corresponding single-language conditions; XLM-R and Qwen were not evaluated in this budget sweep.

\subsection{Domain-Restricted Controls}
\label{subsec:results_domain}

Table~\ref{tab:domain_reference} presents the full nine-condition matrix for BGE-M3+LR and XLM-R, with all eleven pipelines retained in the supplementary tables. Training and validation counts were matched across sources, while the target test memberships were unchanged. On the same KZ-Market target, XLM-R obtained 0.4244 when trained on KZ-Appstore and 0.5141 when trained on KZ-Market. BGE-M3+LR obtained 0.5316 and 0.5442 under those respective source conditions. The difference between training sources was thus not uniform across the two pipelines.

@@DOMAIN_TABLE@@

With Turkish as the fixed target, the KZ-Market-trained models obtained means of 0.7008 for BGE-M3+LR and 0.5800 for XLM-R, compared with 0.6761 and 0.4323 after KZ-Appstore training. These controls document sensitivity to source composition beyond the main language-pair comparison. They do not isolate a causal language effect: the domain labels, review sources, and reference-label distributions were not fully matched across corpora. No transfer-family $p$-value is assigned to these domain-specific differences.

\subsection{Label-Definition Sensitivity}
\label{subsec:results_label_sensitivity}

The binary and extreme-rating analyses used their own task definitions and are therefore reported separately from the three-class results. In the binary task, XLM-R obtained 0.9578 for TR$\rightarrow$TR and 0.7740 for KZ$\rightarrow$KZ; its transfer means were 0.6274 for TR$\rightarrow$KZ and 0.8663 for KZ$\rightarrow$TR. On the Kazakh extreme-rating test view, the same selected binary models obtained 0.6308, 0.7859, and 0.7845 for Turkish, Kazakh, and joint training sources, respectively. The full results for all eleven pipelines are provided in the supplementary tables.

The extreme-rating view retained 31,184--31,316 records per partition from the 34,553-record Kazakh binary test set. No model was retrained on only extreme ratings. Higher scores than in the three-class task reflect a different task and evaluated sample, not a measured correction of label noise. These sensitivity analyses do not establish the independent human validity of the rating-to-sentiment mapping.

\subsection{Class-Level and Representation Diagnostics}
\label{subsec:results_diagnostics}

Neutral-class performance remained weaker than negative- and positive-class performance in the monolingual XLM-R results. For Turkish, the negative, neutral, and positive F1 means were 0.8495, 0.7407, and 0.8562. For Kazakh, they were 0.5866, 0.1815, and 0.8543. Joint XLM-R raised the Kazakh neutral F1 mean to 0.1981, but it remained the lowest of the three class scores. Class-specific values are descriptive and do not inherit the macro-F1 hypothesis-test decisions.

\begin{figure}[htbp]
\centering
\begin{minipage}{0.49\linewidth}\centering
\includegraphics[width=0.94\linewidth]{figures/results_revision/cm_xlmr_tr_to_tr.pdf}
\end{minipage}\hfill
\begin{minipage}{0.49\linewidth}\centering
\includegraphics[width=0.94\linewidth]{figures/results_revision/cm_xlmr_kz_to_kz.pdf}
\end{minipage}
\caption{XLM-R monolingual confusion matrices on the full main test sets. Each cell is the mean of five separately row-normalized matrices (percent); rows are reference classes and columns are predictions. The matrices are descriptive, not counts pooled into an independent five-fold sample. Per-partition reference supports are 10,000/10,000/10,000 for Turkish and 6,031/1,453/28,522 for Kazakh in negative/neutral/positive order.}
\label{fig:xlmr_confusion}
\end{figure}

Figure~\ref{fig:xlmr_confusion} makes the error distributions visible without selecting a best-performing seed. The Kazakh neutral recall was 45.52\% on average, whereas its neutral F1 was much lower; row normalization alone does not display the class prevalence or the false-positive contributions from the much larger positive class. The supplementary class tables retain precision, recall, F1, and supports alongside the matrices.

Lexical diagnostics documented a substantial representational mismatch in direct transfer. In TR$\rightarrow$KZ, the mean fractions of test records mapped to zero vectors were 96.66\% for word TF-IDF and 80.62\% for character TF-IDF. In KZ$\rightarrow$TR, the corresponding fractions were 58.26\% and 0.20\%. Nonzero vectors did not guarantee successful transfer: the latter character-feature coverage coexisted with the lexical scores in Table~\ref{tab:main_results}.

On the common test subset untruncated by all four supervised encoders, the largest absolute change from a full main-task macro-F1 score was 0.00103 across the 330 model--partition evaluations. This is a descriptive test-subset check, not an ablation of tokenization, pretraining, or truncation during training. Length- and script-based slices retain their class supports in the supplementary exports; slices missing a class are flagged rather than interpreted as comparable language-level benchmarks.

\subsection{Supplementary Zero-Shot LLM Results}
\label{subsec:results_llm}

The additional evaluation compared Qwen3-8B with the eleven monolingual supervised references on exactly the same 900-record balanced subset for each language and partition. Table~\ref{tab:llm_balanced_results} is separate from Table~\ref{tab:main_results}: matching test records does not match the amount of task-specific supervision, and balancing the Kazakh subset changes its evaluation distribution.

@@LLM_TABLE@@

Qwen3-8B obtained mean macro-F1 scores of 0.7098 in Turkish and 0.5571 in Kazakh. On those same subsets, BGE-M3+LR obtained 0.7836 and 0.5973, and XLM-R obtained 0.8065 and 0.6005. The mean Qwen-minus-reference differences were $-0.0739$ and $-0.0968$ in Turkish, and $-0.0402$ and $-0.0434$ in Kazakh, for BGE-M3+LR and XLM-R respectively. All twenty partition-specific differences were negative. In the separate 20-test Holm family, both Turkish comparisons rejected in all five partitions; the Kazakh comparisons rejected in three partitions against BGE-M3+LR and two against XLM-R. Non-rejection in the remaining partitions does not establish equivalence.

The neutral class was also Qwen's weakest class. Its mean neutral F1 was 0.5489 in Turkish and 0.4115 in Kazakh, compared with 0.6929 and 0.4932 for BGE-M3+LR and 0.7247 and 0.4986 for XLM-R on the same balanced subsets. Of reference-neutral records, Qwen classified 32.93\% as negative in Turkish and 43.73\% as negative in Kazakh; the corresponding neutral recalls were 47.47\% and 37.00\%. These rates summarize partition-level confusion patterns and do not identify a linguistic cause or validate the reference labels.

All 9,000 evaluation instances were retained. There were 35 invalid responses (0.389\%): one in Turkish and 34 in Kazakh. Of the 34 Kazakh invalid responses, 21 occurred on reference-neutral records. Most neutral errors were nevertheless validly formatted predictions of another class, rather than invalid outputs. Fourteen inputs (0.156\%) were marked as truncated under the fixed Qwen input policy. The aggregate records do not establish why an output was invalid or the causal effect of truncation. These findings apply to the tested model, prompt, decoding configuration, and matched subsets; they do not constitute a comparison of all LLMs or a supervised cross-lingual transfer experiment.
'''
for token,value in [('MAIN_TABLE',t_main),('TRANSFER_TABLE',t_transfer),('JOINT_TABLE',t_joint),('DOMAIN_TABLE',t_domain),('LLM_TABLE',t_llm)]:text=text.replace('@@'+token+'@@',value.strip())
assert '@@' not in text
(ROOT/'SAUCIS_Results_revised.tex').write_text(text,encoding='utf8')

# Supplementary tables retain every pipeline rather than selecting winners.
supp=[r'''% Standalone supplementary Results fragment (not another main Results section).
% The wrapper supplies table/figure prefixes R. All quantities below are
% descriptive summaries unless an explicit Holm-test column is shown.
\section*{Supplementary Results}
The tables retain all evaluated pipelines. Unless otherwise indicated, entries are macro-F1 mean $\pm$ sample SD across five overlapping grouped partitions. The supervised summaries concern the original main-test memberships; the LLM table concerns matched balanced subsets. Machine-readable CSV exports include all 2,820 supervised scores, 564 mean/SD summaries, 330 pointwise main-result intervals, 110 joint paired intervals, 40 supervised transfer tests, 20 exploratory LLM tests, and diagnostic supports. No additional significance test was performed to produce this supplement.
''']
# Low-resource 6 scenario tables, two tables per page.
for idx,(direction,title) in enumerate(zip(DIRECTIONS,DIRNAMES)):
 rows=[[model(r,c)]+[mv('low_mean_sd',f'low:{n}:{direction}',r,c) for n in [100,500,1000,3000,5000]] for r,c in MODELS[:-1]]
 supp.append(table('Low-resource '+title+r'. $n$ is training plus validation records per class and language.','tab:supp_low_'+direction,['Model','$n=100$','$n=500$','$n=1000$','$n=3000$','$n=5000$'],rows,fontsize=r'\footnotesize',colsep='3pt',note='Each single-language task uses $3n$ labelled records; a joint task uses $6n$. XLM-R and Qwen were not part of this budget grid.'))
 if idx%2==1:supp.append(r'\clearpage')
# Domain: all 11 pipelines, 3x3 matrix split by fixed target.
for target in DOMS:
 rows=[[model(r,c)]+[mv('domain_mean_sd','domain:'+src+'_to_'+target,r,c) for src in DOMS] for r,c in MODELS]
 supp.append(table('Domain-restricted evaluation on the fixed '+domlabel[target]+' target. Columns identify the training source.','tab:supp_domain_'+target,['Model']+[domlabel[x] for x in DOMS],rows,fontsize=r'\small',colsep='6pt'))
supp.append(r'\clearpage')
# Binary main and extreme views all models.
for dirs,fn in [(DIRECTIONS[:3],'binary_first'),(DIRECTIONS[3:],'binary_second')]:
 rows=[[model(r,c)]+[mv('binary_mean_sd','binary:'+e,r,c) for e in dirs] for r,c in MODELS]
 hdr=['Model']+[DIRNAMES[DIRECTIONS.index(e)] for e in dirs]
 supp.append(table('Binary-task macro-F1. Neutral records are excluded; binary models are fitted separately from the three-class models.','tab:supp_'+fn,hdr,rows,fontsize=r'\small',colsep='6pt'))
supp.append(r'\clearpage')
rows=[[model(r,c)]+[mv('extreme_mean_sd','extreme:'+e,r,c) for e in ['tr_to_kz','kz_to_kz','joint_to_kz']] for r,c in MODELS]
supp.append(table('Kazakh extreme-rating test view (original rating 1 or 5). Predictions are reused from the selected binary models.','tab:supp_extreme',['Model',DIRNAMES[2],DIRNAMES[1],DIRNAMES[5]],rows,fontsize=r'\small',colsep='6pt'))
# Natural monolingual class results, all 11, two tables.
C=D['all_class_results'];C=C[C.experiment_id.isin(['main:tr_to_tr','main:kz_to_kz'])]
for lang in ['tr','kz']:
 rows=[]
 for r,c in MODELS:
  a=C[(C.experiment_id==f'main:{lang}_to_{lang}')&(C.representation==r)&(C.classifier==c)]
  cells=[model(r,c)]
  for lab in ['negative','neutral','positive']:
   z=a[a.label==lab]['f1-score'];assert len(z)==5;cells.append(pm(z.mean(),z.std(ddof=1)))
  rows.append(cells)
 supp.append(table('Class F1 on the full '+('Turkish' if lang=='tr' else 'Kazakh')+' monolingual test sets.','tab:supp_main_class_'+lang,['Model','Negative','Neutral','Positive'],rows,fontsize=r'\small',colsep='6pt'))
supp.append(r'\clearpage')
# Both exact test families: show every comparison/seed, not only rejections.
for exp,direction in [('main:tr_to_kz',DIRNAMES[2]),('main:kz_to_tr',DIRNAMES[3])]:
 rows=[]
 for key,title in CONTRASTS:
  a=D['transfer_paired_tests_holm40'];a=a[(a.experiment_id==exp)&(a.contrast==key)].sort_values('seed')
  for _,r in a.iterrows():rows.append([title,int(r.seed),f'{r.estimate:+.4f}',f'[{r.ci_low:+.4f}, {r.ci_high:+.4f}]',f'{r.p_raw:.6f}',f'{r.p_holm_40:.6f}'])
 supp.append(table('All partition-specific supervised transfer tests: '+direction+'.','tab:supp_transfer_'+exp.split(':')[1],['Contrast','Seed',r'$\Delta$',r'95\% paired CI','$p$','$p_{\\mathrm{Holm},40}$'],rows,align='lrrrrr',fontsize=r'\footnotesize',colsep='4pt',note='Intervals are pointwise and conditional; $p$-values are adjusted together across both directions and all four contrasts (40 tests). A positive difference favors the first model.'))
 supp.append(r'\clearpage')
for lang in ['tr','kz']:
 rows=[]
 for comparison,short in [('qwen3_8b_minus_bge_m3_lr','Qwen $-$ BGE+LR'),('qwen3_8b_minus_xlmr_finetuned','Qwen $-$ XLM-R')]:
  a=D['llm_exploratory_paired_holm20'];a=a[(a.language==lang)&(a.comparison==comparison)].sort_values('seed')
  for _,r in a.iterrows():rows.append([short,int(r.seed),f'{r.macro_f1_difference:+.4f}',f'[{r.ci_low:+.4f}, {r.ci_high:+.4f}]',f'{r.p_raw:.6f}',f'{r.holm_adjusted_p_20:.6f}'])
 supp.append(table('All exploratory LLM comparisons on the '+('Turkish' if lang=='tr' else 'Kazakh')+' balanced subsets.','tab:supp_llm_tests_'+lang,['Contrast','Seed',r'$\Delta$',r'95\% paired CI','$p$','$p_{\\mathrm{Holm},20}$'],rows,align='lrrrrr',fontsize=r'\footnotesize',colsep='4pt',note='Negative differences favor the supervised reference. The 20-test LLM family is separate from the 40 supervised transfer tests; invalid outputs remain in every calculation.'))
supp.append(r'\clearpage')
for lang in ['tr','kz']:
 rows=[]
 for r,c in MODELS_LLM:
  a=D['llm_class_metrics_mean_sd'];a=a[(a.language==lang)&(a.representation==r)&(a.classifier==c)]
  cells=[model(r,c)]
  for lab in ['negative','neutral','positive']:
   z=a[a.label==lab];assert len(z)==1;cells.append(pm(z.iloc[0].f1_mean,z.iloc[0].f1_sd))
  rows.append(cells)
 supp.append(table('Class F1 on the matched '+('Turkish' if lang=='tr' else 'Kazakh')+' balanced subsets.','tab:supp_llm_class_'+lang,['Model','Negative','Neutral','Positive'],rows,fontsize=r'\small',colsep='6pt',note='These class results are descriptive. Macro-F1 test decisions are not transferred to individual classes.'))
supp.append(r'\clearpage')
# Precise representation coverage, same vocabulary for LR/SVM.
rows=[]
for direction in ['tr_to_kz','kz_to_tr']:
 for rep in ['word','char']:
  d=D['lexical_coverage'];a=d[(d.experiment_id=='main:'+direction)&(d.representation==rep)&(d.classifier=='lr')]
  z=100*a.zero_vector_count/a.n;coverage=100*a.occurrence_coverage
  rows.append([DIRNAMES[DIRECTIONS.index(direction)],NAMES[rep],f'{z.mean():.4f} $\\pm$ {z.std(ddof=1):.4f}',f'{coverage.mean():.4f} $\\pm$ {coverage.std(ddof=1):.4f}'])
supp.append(table('Lexical feature diagnostics in direct transfer (mean $\\pm$ sample SD, percent).','tab:supp_coverage',['Direction','Representation','Zero-vector records','Vocabulary occurrence coverage'],rows,align='llrr',fontsize=r'\footnotesize',colsep='6pt',note='LR records are shown once because LR and SVM share the fitted vectorizer within each source context. Occurrence coverage measures retained analyzer-token occurrences represented in the source vocabulary.'))
supp.append(r'''\paragraph{Additional machine-readable evidence.}
The accompanying \texttt{source\_tables/} directory contains all original aggregate CSV tables and derived descriptive confusion-matrix exports. It retains per-seed accuracy, macro-F1 and weighted-F1; class supports; every individual main-task confidence interval; all joint paired intervals; and diagnostic rows with absent-class flags. Normalized confusion matrices average within-partition row percentages rather than treating overlapping test appearances as independent records. No private text, sample identifier, model weights, or token array is included.

\paragraph{Scope of this supplement.}
Domain, label-definition, class and diagnostic contrasts are descriptive unless explicitly listed in the two hypothesis-test tables. This document does not supply independent human reannotation, causal domain identification, or a comparison with externally published benchmark numbers. Those claims require separate evidence and are not inferred from the current tables.
''')
(ROOT/'SAUCIS_Results_Supplement.tex').write_text('\n\n'.join(supp),encoding='utf8')
print('Generated Results and supplement from aggregate inputs only.')
print('Main subsections:',len(re.findall(r'\\subsection\{',text)))
