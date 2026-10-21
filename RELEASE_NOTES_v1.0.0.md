# v1.0.0 — SAUCIS revision reproducibility package

Research code and aggregate results accompanying:
**Turkish–Kazakh Sentiment Classification: An Empirical Comparison of Lexical and Multilingual Models**.

## Included
- Twelve frozen stage scripts covering source auditing, grouped partitions,
  lexical and embedding models, convergence recovery, XLM-R fine-tuning, final
  evaluation, conditional statistical analysis and Qwen3-8B zero-shot evaluation.
- Pinned upstream input specifications, original code hashes, recorded supervised
  package versions, supplementary methods and the exact Qwen prompt.
- Public evidence for 2,820 supervised evaluations, 564 summaries, 330 conditional
  main-score intervals, 110 joint differences, 40 supervised comparisons and a
  separate supplementary LLM set with 120 result rows and 20 paired comparisons.
- An offline aggregate audit, synthetic tests and publication figure source data.

## Excluded
Raw reviews, per-record predictions/identities, local SQLite databases, tokens,
private logs, trained/pretrained model weights and unrelated project content.

## Reproduction statement
Original experiment scripts are byte-identical. New release helpers are explicitly
separate. Successful original runs are documented by aggregate artifacts. Release
validation recomputes aggregate arithmetic and runs synthetic tests; it does not
claim a new full GPU rerun. The step09 summary-counter erratum and metadata
redactions are documented without changing scientific results.

No publication acceptance or article DOI is asserted. The software's version DOI
is assigned by Zenodo after the owner publishes the release.
