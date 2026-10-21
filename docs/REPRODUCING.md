# Reproducing the study

## A. Inspect the package without re-running the experiments

From the extracted repository root:

```bash
python tools/audit_release.py
```

This is a standard-library, offline consistency audit. No installation, GPU,
model loading or external account is required. The reports and tables are sufficient
to audit classification arithmetic, mean/SD, contrast points and reported probability
corrections. They do not contain the private group-level predictions needed to
repeat bootstrap draws. A full reproduction below regenerates those privately.

## B. Install the recorded supervised environment

The research pipeline uses Linux file locking and NVIDIA CUDA. Python **3.12.3**
was recorded. Use a separate virtual environment for reproduction.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r environment/requirements-supervised.txt
```

Verify the actual interpreter is 3.12.3 and the installed torch build is
2.10.0+cu128 before claiming a version-matched reproduction. These commands are
an installation recipe, not evidence that a fresh CUDA environment was installed
or tested during release packaging. Drivers and GPU allocation are managed by
the local system administrator. Dependencies are recorded direct packages, not
a complete lock of every transitive wheel.

The original shared Qwen record does not independently list its full package
versions. Save a new environment inventory for any reproduction rather than
claiming the supervised inventory was a separate Qwen capture.

## C. Obtain the exact inputs in an isolated workspace

Remain in the release root. The following command downloads the two pinned source
CSVs and exactly the audited files for the four supervised encoders. It copies
the unchanged experiment scripts into `_reproduction/` and writes a **new**, clearly
identified input inventory. It never searches unrelated repositories or user files.

```bash
python tools/prepare_inputs.py \
  --workspace ./_reproduction \
  --assets supervised \
  --allow-download
```

Omit `--allow-download` for a cache-only check. The helper refuses different CSV
checksums, model file checksums or a conflicting isolated `refs/main`. It preserves
the original BGE-M3 weight format rather than silently replacing it with another
revision. Dataset/model licenses remain the upstream providers' terms. Access
credentials, when needed, are handled through local Hugging Face authentication;
never insert them into repository files.

Step00 records the original environment diagnostics.
It need not be run on a new machine: it can discover multiple copies of known
caches and lacks prepared-dataset metadata on a clean installation. The release
adapter instead emits only the explicitly selected inputs. The provided reference
row counts are labeled as coming from the historical audit for identical bytes;
step01 still reads the actual rows and step02 still requires its genuine checks.

Allow substantial free disk space. The full XLM-R run retains 270 epoch weight
files (the historical files were approximately 1.1 GB each), in addition to feature
banks, predictions, source models and rotating full-state checkpoints. Check free
space for the actual workspace filesystem; do not delete completed checkpoints
or disable safety checks to force progress. Runtime depends on the available hardware.

## D. Run one stage at a time

`tools/run_stage.py` runs exactly one selected stage. Without `--execute` it only
prints a command. It records the actual output paths under the workspace, because
fresh audit/protocol identifiers can differ from the paper's historical identifiers.
It does not select a path by result score, skip unsuccessful candidates, or start
the next stage automatically. Each underlying original stage retains its own
checksum/completion gates. Do not rename new artifacts to historical folder names.

```bash
python tools/run_stage.py --workspace ./_reproduction --stage audit --execute
python tools/run_stage.py --workspace ./_reproduction --stage split --execute
```

Only use a GPU explicitly allocated to you. Replace `1` below with that physical
index or full UUID; the wrapper preserves inherited device restrictions.

```bash
python tools/run_stage.py --workspace ./_reproduction --stage smoke --gpu 1 --execute
python tools/run_stage.py --workspace ./_reproduction --stage features --gpu 1 --execute
python tools/run_stage.py --workspace ./_reproduction --stage linear --execute
```

The original step05 may stop at a nonconverged SVM candidate. Preserve its output.
The recovery implementation imports valid completed candidates and retries convergence
failures with the same settings except for the documented iteration ceiling:

```bash
python tools/run_stage.py --workspace ./_reproduction --stage recovery --execute
```

Proceed only after `LINEAR_MODELS_READY`. Other errors must be investigated rather
than assumed to be convergence failures.

```bash
python tools/run_stage.py --workspace ./_reproduction --stage pilot --gpu 1 --execute
python tools/run_stage.py --workspace ./_reproduction --stage xlmr --gpu 1 --execute
python tools/run_stage.py --workspace ./_reproduction --stage test --gpu 1 --execute
python tools/run_stage.py --workspace ./_reproduction --stage statistics --execute
```

Expected scientific coverage is 6,000 converged C fits / 1,200 linear selections;
90 XLM-R learning-rate candidates / 45 selections / 270 epoch validations;
2,820 supervised test evaluations; 330 single-score CIs / 40 primary contrasts /
110 joint paired CIs. The known historical step09 counter field is documented
separately. Logs are not evidence of a stage completing unless its own state says so.

For an interruption, rerun that same stage with unchanged scientific settings.
The original code validates committed work. Changing a GPU is recorded, not
claimed bitwise identical. Do not mix partial artifacts from different code hashes,
protocols or environments. A single Ctrl+C requests the original worker's controlled
exit; allow its checkpoint operation to finish instead of killing it mid-write.

## E. Supplementary Qwen3-8B run with the ORIGINAL snapshot

```bash
python tools/run_stage.py --workspace ./_reproduction --stage llm-prepare --execute
python tools/run_stage.py --workspace ./_reproduction --stage qwen-pin --allow-download --execute
python tools/run_stage.py --workspace ./_reproduction --stage llm-pilot --gpu 1 --execute
python tools/run_stage.py --workspace ./_reproduction --stage llm-run --gpu 1 --execute
python tools/run_stage.py --workspace ./_reproduction --stage llm-analyze --execute
```

The release helper pins Qwen3-8B to
`b968826d9c46dd6066d109eabc6255188de91218`. Do NOT substitute the original stage10's
first-time `download --allow-download` branch for `qwen-pin`: that historical
branch resolves upstream main at first use. The model worker, prompt, generation
settings, sampling policy and metric calculations themselves remain unchanged.

The Qwen pilot uses validation text for engineering checks, not label-based prompt
tuning. It is not a model-quality estimate. The 9,000 balanced-subset evaluation
appearances remain a separate supplementary study and do not replace the main
natural-distribution results. Keep invalid responses in the metrics.

## F. Figure recreation

The four original publication figures and numeric inputs are already supplied.
Optional recreation (no model inference or new hypothesis tests):

```bash
python -m pip install -r environment/requirements-figures.txt
python paper_outputs/reproduce/build_results_figures.py
```

This overwrites the **figure copies in this release tree**, not research outputs;
make a working copy first when preserving the downloaded archive's manifest.

## G. Synthetic tests

```bash
python -m pip install -r environment/requirements-tests.txt
python -m pytest -q tests
```

The original LLM unit tests use fabricated arrays/tokenizer responses. New release
checks exercise command construction, path safeguards, pinned-file validation,
metadata selection and aggregate arithmetic. Passing these tests does not certify
full GPU retraining, private-row resampling, label validity or future performance.

## Preserve completed upstream identities

After downstream stages have begun, do not regenerate an upstream audit merely
to refresh its timestamp. Historical stage identities include report hashes; a
new upstream report may therefore create a different downstream identity even
with the same input records. Use the recorded workspace stage paths and resume
the interrupted stage. Never rename a fresh run to impersonate the archived
protocol identifier.
