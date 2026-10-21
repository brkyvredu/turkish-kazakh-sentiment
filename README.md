# Turkish–Kazakh sentiment classification

Reproducibility source package for **Turkish–Kazakh Sentiment Classification:
An Empirical Comparison of Lexical and Multilingual Models** — Ismail Burak Yavru,
Bursa Uludag University.

**GitHub repository:** [https://github.com/brkyvredu/turkish-kazakh-sentiment](https://github.com/brkyvredu/turkish-kazakh-sentiment)

**Prepared version:** `1.0.0`. The owner publishes the GitHub release and Zenodo
record. This archive itself creates neither a repository nor a DOI. Before the
first public release, complete the explicit license/metadata step in
[`YAYINLA_TR.md`](YAYINLA_TR.md). No article acceptance or article DOI is asserted.

## Start here: verify the distributed evidence without training

```bash
python tools/audit_release.py
```

This command uses the Python standard library, works offline, and does not load
models. It verifies original source hashes; recomputes metrics from aggregate
confusion matrices, five-partition means and sample SDs, contrast points, Monte
Carlo plus-one probabilities and Holm corrections; and checks exclusion of private
runtime artifacts. It does **not** rerun bootstrap/permutation draws, model training,
or inference. The packaged validation report states exactly which checks ran.

## Contents

| Location | Contents |
| --- | --- |
| `00_revision_*.py` through `10_revision_*.py` | Twelve byte-identical original stage scripts, including both step05 and its convergence-recovery successor |
| `metadata/frozen_scripts.json` | Original code sizes and SHA-256 values |
| `metadata/upstream_assets.json` | Dataset revisions/checksums and model pins; historical per-file model hashes where available |
| `metadata/experiment_configuration.json` | Constants extracted from the unchanged scripts, including the exact Qwen prompt |
| `results/01_data_audit/` … `results/10_llm/` | Complete, successful-run public aggregates and manifests; large JSON reports are losslessly gzip-compressed |
| `paper_outputs/source_tables/` | Tables used for manuscript reporting and descriptive figure inputs |
| `paper_outputs/reproduce/` | Original publication figure generator |
| `docs/supplementary_methods/` | Model snapshots, Qwen prompt and statistical implementation supplementary methods |
| `tools/` | New release-only input preparation, metadata and offline audit helpers |
| `tests/` | Synthetic/unit checks; no private dataset is used |
| `provenance/` | File origins, original/public hashes, export redactions and current validation results |

No raw review text, individual sample/group identities, private predictions,
training checkpoint, pretrained model weight, credentials, machine cache or
unrelated project code is included. Files referring to `PRIVATE` in their source
code are producers/readers of local artifacts, not embedded private data.

## Evaluation scope

The main study uses TRSAv1 and KazSAnDRA with five grouped partitions (seeds 42–46).
Ten lexical/frozen-embedding pipelines keep LR and SVM separate; XLM-R is fully
fine-tuned. Main, binary, extreme-rating, low-budget and domain conditions yield
2,820 supervised evaluations and 564 five-partition summaries. Model selection
uses source validation; the test results are not used to choose a representation's
best classifier. Training-only class capping does not rebalance validation/test.

Step09 contains 330 main-score conditional intervals, 40 paired supervised
transfer comparisons, and 110 joint-minus-monolingual intervals. The separate
Qwen3-8B zero-shot experiment contains 120 matched-subset result rows and 20
exploratory paired comparisons. Its 9,000 evaluation appearances can overlap
across partitions. It matches test records, not training-label budgets.

The 40-test and 20-test Holm families remain separate. Five partitions are not
five independent external corpora. Intervals are pointwise and conditional on
fitted models and an evaluated partition; no universal model ranking or independent
human validation of rating-derived labels is asserted.

## Two distinct uses

1. **Inspect and audit the published aggregates:** use `tools/audit_release.py`.
   This does not need GPUs, the original private runtime outputs, or an Internet
   connection.
2. **Reproduce from pinned upstream inputs:** follow
   [`docs/REPRODUCING.md`](docs/REPRODUCING.md) in a fresh workspace. Data/model
   downloads and long training runs require explicit user action. Do not use the
   public aggregate folders as if they were completed private runtime directories.

The public bundle is not a trained-model inference service. Retraining/inference
will generate private artifacts locally. Results from another hardware/software
stack are not promised to be bitwise identical. Full GPU retraining was **not**
performed during release packaging; source provenance and numerical aggregate
checks are not substitutes for that experiment.

## Environment

The supervised records specify Python 3.12.3, NumPy 2.3.5, scikit-learn 1.8.0,
PyTorch 2.10.0+cu128, Transformers 5.12.1 and Sentence Transformers 5.6.0.
Exact recorded packages are under `environment/`. These are observed versions,
not a claim that every dependency combination was freshly installed or resolved.
The shared Qwen report does not provide a separate complete dependency snapshot;
see `docs/REPRODUCING.md` for that limitation.

## Source integrity and a known reporting erratum

The twelve experiment scripts are intentionally unchanged: their SHA-256 values
are used in artifact identities and cross-script checks. In historical step09,
`completed_analysis_units` remains at 788 even though 810 units finished. All 110
joint rows are present, including the last 22. The counter is updated before the
last joint loop and not refreshed at exit. This is documented in
[`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md); no score, interval or p-value is
silently altered. Public JSON exports redact local home paths and workstation
UUIDs; provenance records distinguish their hashes from the historical originals.

## Citation and licensing

`CITATION.cff` identifies this software. Add the actual version-specific Zenodo DOI
after publication; no placeholder DOI is presented as real. The manuscript title
identifies the associated work without implying publication. A license option is
provided but is not activated on the author's behalf. See
[`LICENSE_CHOICE_REQUIRED.md`](LICENSE_CHOICE_REQUIRED.md). Upstream datasets,
models and dependencies retain their own terms and are not redistributed here.

## Documentation

- [`YAYINLA_TR.md`](YAYINLA_TR.md): GitHub/Zenodo publishing checklist for the owner.
- [`docs/REPRODUCING.md`](docs/REPRODUCING.md): stage commands and pinned inputs.
- [`docs/DATA_AND_RESULTS.md`](docs/DATA_AND_RESULTS.md): public result schema and exclusions.
- [`docs/KNOWN_ISSUES.md`](docs/KNOWN_ISSUES.md): documented reproducibility boundaries.
- [`RELEASE_NOTES_v1.0.0.md`](RELEASE_NOTES_v1.0.0.md): proposed release description.
