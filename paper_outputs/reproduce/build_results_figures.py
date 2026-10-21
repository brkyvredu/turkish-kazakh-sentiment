#!/usr/bin/env python3
"""Create four separate publication figures from completed aggregate results.
No inference, training or statistical resampling. Every figure uses one Axes.
Blue/purple/grey/orange palette follows the earlier manuscript color constraint.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'source_tables';OUT=ROOT/'figures/results_revision';OUT.mkdir(parents=True,exist_ok=True)
low=pd.read_csv(DATA/'low_mean_sd.csv')
cm=pd.read_csv(DATA/'main_confusion_rowpercent_mean_sd.csv')
repnames={'word':'Word TF-IDF','char':'Char TF-IDF','minilm':'MiniLM','labse':'LaBSE','bge_m3':'BGE-M3'}
# No red/yellow/green combination is used.
palette={'word':'#5A626C','char':'#8A5B3D','minilm':'#7461A8','labse':'#497EA8','bge_m3':'#152E53'}
markers={'word':'o','char':'s','minilm':'^','labse':'D','bge_m3':'P'}
styles={'word':'--','char':'-.','minilm':':','labse':'--','bge_m3':'-'}
figure_trace=[]
for lang,label in [('tr','(a) Turkish: TR  TR'),('kz','(b) Kazakh: KZ  KZ')]:
 label=label.replace('\u001a','\N{RIGHTWARDS ARROW}')
 fig=plt.figure(figsize=(4.15,3.05));ax=fig.add_subplot(111)
 for rep in ['word','char','minilm','labse','bge_m3']:
  part=low[(low.representation==rep)&(low.classifier=='lr') & low.experiment_id.str.match(rf'low:\d+:{lang}_to_{lang}$')].copy()
  part['budget']=part.experiment_id.str.split(':').str[1].astype(int);part=part.sort_values('budget')
  assert part.budget.tolist()==[100,500,1000,3000,5000] and part.n_seeds.eq(5).all()
  ax.errorbar(part.budget,part.macro_f1_mean,yerr=part.macro_f1_sd,label=repnames[rep],color=palette[rep],marker=markers[rep],linestyle=styles[rep],lw=1.35,ms=4.2,capsize=2,elinewidth=.8)
 ax.set_xscale('log');ax.xaxis.set_major_locator(FixedLocator([100,500,1000,3000,5000]));ax.xaxis.set_major_formatter(FixedFormatter(['100','500','1000','3000','5000']));ax.xaxis.set_minor_locator(NullLocator())
 ax.set_xlim(83,6100);ax.set_ylim((.58,.82) if lang=='tr' else (.35,.58))
 ax.set_xlabel('Total labels per class and language, n',fontsize=10.5)
 ax.set_ylabel('Macro-F1',fontsize=10.5)
 ax.set_title(label,fontsize=11,pad=10)
 ax.tick_params(labelsize=9.5);ax.spines[['top','right']].set_visible(False)
 ax.grid(axis='y',alpha=.18,lw=.6)
 ax.legend(loc='lower right',ncol=2,fontsize=8.5,frameon=False,columnspacing=.8,handlelength=1.8,borderpad=.15)
 fig.tight_layout(pad=.8)
 name=f'low_{lang}_to_{lang}_lr'
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
 figure_trace.append(dict(file=name+'.pdf',source='low_mean_sd.csv',filter=dict(classifier='lr',scenario=f'{lang}_to_{lang}'),series=list(repnames),aggregation='existing five-seed mean and sample SD; no confidence interval'))
for lang,label in [('tr','(a) XLM-R: TR  TR'),('kz','(b) XLM-R: KZ  KZ')]:
 label=label.replace('\u001a','\N{RIGHTWARDS ARROW}')
 part=cm[(cm.representation=='xlmr')&(cm.classifier=='finetuned')&(cm.experiment_id==f'main:{lang}_to_{lang}')]
 labs=['negative','neutral','positive'];matrix=part.pivot(index='reference_label',columns='predicted_label',values='row_percent_mean').reindex(index=labs,columns=labs).to_numpy()
 assert matrix.shape==(3,3) and np.allclose(matrix.sum(1),100)
 fig=plt.figure(figsize=(4.15,3.35));ax=fig.add_subplot(111)
 im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=100,interpolation='nearest')
 ax.set_xticks(range(3),['Negative','Neutral','Positive']);ax.set_yticks(range(3),['Negative','Neutral','Positive'])
 ax.tick_params(length=0,labelsize=8.7,pad=6)
 for i in range(3):
  for j in range(3):ax.text(j,i,f'{matrix[i,j]:.2f}',ha='center',va='center',fontsize=10,color='white' if matrix[i,j]>=55 else '#162435')
 ax.set_xlabel('Predicted class',fontsize=10.5);ax.set_ylabel('Reference class',fontsize=10.5);ax.set_title(label,fontsize=10.3,pad=12)
 cb=fig.colorbar(im,ax=ax,fraction=.042,pad=.055);cb.set_ticks([0,50,100]);cb.ax.tick_params(labelsize=8);cb.set_label('Row %',fontsize=8.5)
 fig.tight_layout(pad=.7)
 name=f'cm_xlmr_{lang}_to_{lang}'
 fig.savefig(OUT/(name+'.pdf'),bbox_inches='tight');fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight');plt.close(fig)
 figure_trace.append(dict(file=name+'.pdf',source='main_confusion_rowpercent_mean_sd.csv',filter=dict(representation='xlmr',classifier='finetuned',experiment_id=f'main:{lang}_to_{lang}'),aggregation='normalize each seed confusion row first, then arithmetic mean of five; no pooling or best-seed selection'))
(ROOT/'qa/figure_provenance.json').write_text(json.dumps(figure_trace,indent=2),encoding='utf8')
print('Created 4 separate vector-PDF figures and PNG previews.')
