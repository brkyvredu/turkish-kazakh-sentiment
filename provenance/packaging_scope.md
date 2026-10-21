# Source and artifact provenance

The twelve experimental scripts are preserved byte-for-byte.
`metadata/frozen_scripts.json` records their file sizes and SHA-256 checksums.
The aggregate results come from the completed experimental runs; interrupted
runs and private runtime files are excluded.

Input-preparation, execution and metadata utilities under `tools/` were added
for distribution. Their tests use synthetic data and command/schema checks.
They were not used to generate the original training results, and release
validation does not constitute a new GPU reproduction.

Public report copies redact absolute home paths and workstation identifiers.
Original and distributed hashes are recorded in `provenance/result_sources.json`.
The numerical results are unchanged. Large JSON reports use lossless gzip compression.

The LLM tests include two filesystem-path changes for the repository layout;
the corresponding diff is included. Validation outputs distinguish these tests
from the original experimental runs.

AI-assisted tools supported code and documentation development. The named author
is responsible for the research, verification and release.
