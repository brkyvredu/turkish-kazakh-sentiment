# v1.0.0 — Turkish–Kazakh sentiment classification

Research code and aggregate results for **Turkish–Kazakh Sentiment Classification:
An Empirical Comparison of Lexical and Multilingual Models**.

## Contents

- Twelve experimental scripts covering data audits, grouped partitions, lexical
  and embedding models, convergence recovery, XLM-R training, test evaluation,
  statistical analysis and zero-shot Qwen3-8B evaluation.
- Pinned inputs, recorded dependencies, supplementary methods and the fixed Qwen prompt.
- 2,820 supervised results, 564 five-partition summaries, 330 main-score intervals,
  110 joint-training differences and 40 paired transfer comparisons.
- A separate LLM evaluation with 120 result rows and 20 paired comparisons.
- Offline validation tools, unit tests and figure source data.

## Reproduction

The experimental scripts are preserved byte-for-byte. Release validation checks
aggregate arithmetic and source integrity; it does not repeat GPU experiments or
group-level resampling. The historical step09 counter discrepancy is documented
in `docs/KNOWN_ISSUES.md` and does not affect the result tables.

Raw review texts, per-record predictions, local databases and model weights are
not included. Input retrieval and execution instructions are in `docs/REPRODUCING.md`.

## License

MIT. Third-party datasets, models and dependencies retain their own licenses.
