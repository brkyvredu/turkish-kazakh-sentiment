# Data and results dictionary

## Upstream inputs (not redistributed)
`metadata/upstream_assets.json` records the original TRSAv1 and KazSAnDRA commit
identities, original CSV filenames, byte counts and SHA-256 values. The same file
lists the actual MiniLM, LaBSE, BGE-M3, XLM-R and Qwen model revisions. Model weight
files are downloaded separately under the upstream providers' applicable terms.
The source datasets' reference labels are preserved; contradictory normalized-text
labels are not resolved by vote. No independent human reannotation is added.

## Successful historical outputs
Only final completed stage archives are included. Earlier blocked training files
and development error logs are not distributed.

- `results/01_data_audit/tables/`: raw/eligible record flows, class/domain counts,
  text-length and script summaries. These are aggregate descriptions, not raw text.
- `results/02_splits/`: protocol, per-seed aggregate counts and experiment recipes.
  The `*.local.sqlite3` membership/corpus databases are NOT included.
- `results/05_linear/`: the full validation C-grid results and selected LR/SVM
  settings. Successful recovery is documented; it does not imply test selection.
- `results/07_xlmr/`: 270 epoch-validation rows and 45 validation selections.
- `results/08_test/all_test_results.csv`: 2,820 rows keyed by seed, experiment,
  representation and classifier; accuracy, macro-F1 and weighted-F1 are per-run
  test metrics. `class_test_results.csv` retains precision, recall, F1 and support.
- `results/08_test/test_evaluation_report.json.gz`: full aggregate test report,
  including confusion matrices; lossless gzip after documented metadata redaction.
- `results/09_analysis/tables/`: five-seed descriptive summaries, 330 main-score
  conditional CIs, 40 transfer tests, 110 joint differences and diagnostic slices.
- `results/10_llm/`: 120 matched-balanced-subset rows, 24 five-partition summaries,
  20 exploratory paired tests and class confusion matrices, including abstentions.

`macro_f1_sd` uses the sample SD across five overlapping partitions; it is not a
confidence interval. Primary paired differences have direction A minus B. LLM
contrasts are Qwen minus the supervised reference. The 40 and 20 Holm families
are separate; their p-values must not be transferred to untested class/domain
contrasts. A row's test count is not a new independent corpus.

LLM confusion matrices have three reference rows and four prediction columns:
NEGATIVE, NEUTRAL, POSITIVE, INVALID. Invalid outputs contribute false negatives
for their reference class and remain in the evaluated total. The extra column is
not a fourth reference sentiment class.

## Statistical replay versus arithmetic audit
Aggregate confusion matrices can reproduce classification metrics; raw extrema
counts can reproduce plus-one p-values and Holm adjustments; saved per-seed rows
can reproduce means/SDs and contrast points. They cannot alone rerun group-level
resampling: that requires the private row/group assignments regenerated during a
full local reproduction. No per-record identities or review text are published.

## Paper assets
`paper_outputs/source_tables/` mirrors the manuscript's numeric aggregate inputs.
The four PDF/PNG figures and their generator are included. These are stored figure
artifacts; no full manuscript, reviewer report, private mail, or similarity report
is bundled. Supplementary methods are included as a separate document under `docs/`.
