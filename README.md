# Turkish–Kazakh Sentiment Classification

Research code and aggregate results for **Turkish–Kazakh Sentiment Classification:
An Empirical Comparison of Lexical and Multilingual Models**.

**Author:** Ismail Burak Yavru, Bursa Uludag University  
**Version:** 1.0.0 · **License:** [MIT](LICENSE)

## Overview

The study compares word and character TF-IDF, frozen MiniLM, LaBSE and BGE-M3
embeddings, and fine-tuned XLM-R on TRSAv1 and KazSAnDRA. Five grouped partitions
support monolingual, cross-lingual and joint-training evaluations, supplemented by
label-budget, domain and class-error analyses. Qwen3-8B is evaluated separately
in a zero-shot setting on matched balanced subsets.

Logistic regression and linear SVM results are reported separately. Model selection
uses validation data; transferred models use source-language validation. The release
contains 2,820 supervised evaluation results, 564 five-partition summaries, and
120 results from the separate LLM comparison.

## Verify the results

Run from the repository root:

```bash
python tools/audit_release.py
```

This offline check uses the Python standard library. It verifies the experimental
script hashes and recomputes aggregate metrics, means, standard deviations, paired
differences and Holm corrections. It does not rerun model training, inference or
bootstrap/randomization sampling.

## Reproduce the experiments

See [Reproducing the study](docs/REPRODUCING.md) for installation, pinned input
retrieval and stage-by-stage commands. Full execution requires Linux or WSL and an
NVIDIA GPU for the encoder and language-model stages. The recorded supervised
environment uses Python 3.12.3; dependency records are in [`environment/`](environment/).

Raw reviews, individual predictions, partition databases and model weights are
not distributed. They are obtained or generated locally when reproducing the
experiments. Dataset and model revisions are listed in
[`metadata/upstream_assets.json`](metadata/upstream_assets.json).

## Repository contents

| Path | Contents |
| --- | --- |
| `00_revision_*.py`–`10_revision_*.py` | Experimental pipeline, including the step05 convergence-recovery implementation |
| [`results/`](results/) | Aggregate data audits, model selections, evaluation results and statistical analyses |
| [`metadata/`](metadata/) | Experimental configuration, model/data revisions and script checksums |
| [`paper_outputs/`](paper_outputs/) | Figure scripts and the aggregate tables used in the manuscript |
| [`docs/supplementary_methods/`](docs/supplementary_methods/) | Model snapshots, fixed Qwen prompt and supplementary methods |
| [`tools/`](tools/) | Input preparation, execution and validation utilities |
| [`tests/`](tests/) | Unit and synthetic-data tests |
| [`provenance/`](provenance/) | Artifact origins, export transformations and validation records |

## Interpreting the results

The five partitions share the same two source corpora. Their test records may
overlap, so they are not independent external datasets. Confidence intervals are
conditional on the fitted models and evaluated partition. The 40 supervised
transfer comparisons and 20 exploratory LLM comparisons use separate Holm families.
The balanced LLM subsets match test records, not supervision budgets.

The experimental scripts are retained unchanged. Known implementation and
reproduction limitations, including the step09 summary-counter discrepancy, are
listed in [Known issues](docs/KNOWN_ISSUES.md). The result schemas are described in
[Data and results](docs/DATA_AND_RESULTS.md).

## Citation

Citation metadata is available in [`CITATION.cff`](CITATION.cff).

## License

The code and accompanying documentation are released under the [MIT License](LICENSE).
Third-party datasets, pretrained models and dependencies remain subject to their
respective licenses.
