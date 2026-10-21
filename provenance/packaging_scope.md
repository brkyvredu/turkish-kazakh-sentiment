# Packaging provenance

The twelve original experiment Python files were recovered from the conversation
and saved Library artifacts. They are copied byte-for-byte; `metadata/frozen_scripts.json`
records their identities. Earlier interrupted output archives are excluded.

Release-only helpers and documentation were prepared after the reported experiments.
They are not claimed to have produced the original training or inference results.
AI-assisted coding/documentation support was used in this project and packaging;
scientific authorship, interpretation, licensing and final release approval remain
with the named author. No publisher compliance assertion is inferred from packaging.

Original output member hashes are retained alongside the public hashes. Redactions
remove absolute home paths and workstation UUIDs only. Numeric values and model
selection records are unchanged. Some large aggregate JSON files are gzip-compressed
so the owner need not upload very large individual files through the GitHub browser.

The 45 existing LLM synthetic tests are included with only two filesystem-path
adjustments for this release layout; their exact diff is supplied. New release tests
are labeled separately. Current test outputs do not stand in for original GPU logs.
