# Known issues and reproduction boundaries

## Historical step09 summary counter
The completed historical run reports `new_units=810`, `reused_units=0`, but
`completed_analysis_units=788`. The last field is updated before the final
partition's 22 joint units and not refreshed in `finally`. The output contains
330 single-score intervals, 330 diagnostic units, 40 primary pairs and 110 joint
pairs (810 total). All final-partition joint rows are present. No statistics are
missing for this reason. The frozen script is preserved rather than modified
under the same experiment identity. A metadata erratum is shipped separately.

## Dynamic run directory identities
Step02 includes the data-audit report hash (and therefore fresh audit provenance)
in its protocol identity. New clean runs do not have to use the historical
`dc483a2f7daabc096c06` folder name even with identical raw CSV files. Pass the actual
`[OUTPUT]` paths between stages. Never rename fresh artifacts to pretend they came
from the historical run. SQLite checksums also depend on creation/runtime details.

## Original step10 download resolves upstream main on its first download
For archival fidelity, the original code is retained. For NEW reproduction, do
not use its initial `--stage download --allow-download` path, which resolves the
then-current main revision. Instead use `tools/prepare_inputs.py --assets qwen`
after `--stage prepare`: the helper pins the exact original Qwen commit and creates
the input manifest that the unchanged worker verifies. It does not change prompt,
sampling, decoding, precision, or model parameters.

## Original step00 scans known caches
The original preflight is an author-environment diagnostic. For a clean workspace,
use the release input adapter, which inventories only the chosen pinned cache and
supplies explicitly identified historical row-count references. This avoids
ambiguous multiple cache copies and absent prepared-dataset metadata. Step01 still
reads and audits all raw rows; no PASS result is manufactured by the adapter.

## Step05 convergence recovery
Both original step05 and its recovery implementation are included. The recovery
CLI requires `--resume-from` the actual step05 directory, even if step05 is blocked
on an iteration limit. Completed candidate artifacts must not be deleted. Recovery
changes the maximum iterations only after convergence failure, retaining the
scientific grid and validation rules. A different environment may fail for other
reasons: do not relabel such failures as convergence or skip unsuccessful fits.

## Portability and testing
The research scripts use Linux file locks and NVIDIA CUDA where indicated. Windows
users should use Linux/WSL for full execution. The public audit itself is stdlib-only.
The original trained models and row-level predictions are absent by design. Current
release checks validate code identity and aggregate arithmetic; they do not replay
GPU training/inference or private bootstrap draws. Hardware changes can affect
floating-point results even with fixed seeds.

## Privacy transformation of public reports
A few aggregate JSON exports contain workstation UUIDs or home paths. These strings
are redacted in the public copy. Original-member hashes and published-content hashes
are both retained in `provenance/result_sources.json`. CSV numerical evidence is
unchanged. Public reports are evidence, not resumable runtime inputs; descriptors
of excluded private files do not mean those files are distributed.

## Qwen environment capture
The supplied aggregate Qwen report records the model commit, GPU class, run counts,
invalid outputs, prompt policy and results, but not a standalone full dependency
inventory. The supervised environment file must not be described as an independently
captured Qwen lockfile. Reproduction users should save their actual environment.

## Execution helpers are new release code
`prepare_inputs.py` and `run_stage.py` are tested with local synthetic checks and
command/schema checks, not a new full download-to-GPU end-to-end run. The original
pipeline code is unchanged, and unexpected incompatibilities must stop a run
rather than be repaired by silently changing experimental settings.
