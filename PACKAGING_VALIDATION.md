# Packaging validation

Prepared: 2026-11-03. Package identifier: `2026.11.03-prl5`.

## Completed checks

- Verified all 285 checksummed entries in the source reproducibility archive.
- Compared all 283 files in `scientific_code/` and `prl_figures/` byte-for-byte
  with that archive. Both trees are unchanged.
- Verified the original scientific manifest: all 266 entries passed, with no
  missing, changed, or unlisted files.
- Parsed all 71 Python source files and all 75 JSON files successfully.
- Listed all 57 runner tasks without executing calculations and verified that
  every named task script exists.
- Ran the three existing `test_output_paths.py` tests: all passed. These check
  relative path handling and separation of reference data from fresh results.
- Confirmed the five current PRL figure PDFs match the reference outputs in this
  package. The manuscript and its LaTeX sources are excluded.
- Inspected the supplied trees for private review correspondence, credential
  markers, personal absolute paths, and unsafe archive paths or symlinks;
  no such material was found. Scientific validation logs and reference plots
  remain included.

The source archive is identified in [`provenance/README.md`](provenance/README.md).
The root `MANIFEST.sha256` covers this complete package, including the unchanged
scientific manifest. Use the root README's verification commands after unpacking.

## Scope

Packaging and path tests used Python 3.15.2; this is not a new
numerical validation environment. The recorded scientific validation environment
is Python 3.14.3 with the versions in `scientific_code/requirements.txt`.
No numerical or symbolic calculations, figure regeneration, or complete runner
regression suite were rerun for this packaging step. Historical validation
records retain their original dates and scope; see
[`scientific_code/docs/VALIDATION.md`](scientific_code/docs/VALIDATION.md).
