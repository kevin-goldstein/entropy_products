# Entropy products beyond the attractor

Scientific code, reference data, and figure scripts accompanying the paper by
Kevin Goldstein, Vishnu Jejjala, and Suresh Nampuri.

The calculations investigate conserved black-hole entropy products, deficits
and excesses relative to the extremal reference, fixed-coupling families,
scalar focusing, and entropy-gap bounds.

This repository package contains code and data only. The manuscript and its
LaTeX sources are distributed separately. Reference figure PDFs are included.

## Repository layout

| Path | Contents |
| --- | --- |
| [`scientific_code/`](scientific_code/) | Preserved scientific snapshot `2026.11.02-r5`: calculations, controls, tests, data, and validation records. |
| [`prl_figures/`](prl_figures/) | PRL figure scripts, their input data and metadata, and reference figure PDFs. |
| [`REPRODUCING_PRL.txt`](REPRODUCING_PRL.txt) | Current Letter/Supplement figure and table map, with reproduction commands. |
| [`PACKAGING_VALIDATION.md`](PACKAGING_VALIDATION.md) | Checks performed on this package and their scope. |
| [`provenance/`](provenance/) | Metadata and checksums identifying the source reproducibility archive. |

The scientific snapshot is preserved byte-for-byte, including its original
title and documentation. Its internal paper map and figure numbering refer to
the earlier long manuscript. Use the top-level
[`REPRODUCING_PRL.txt`](REPRODUCING_PRL.txt) for the current PRL figure and table map.

## Setup

The recorded numerical validation environment uses **Python 3.14.3**, with exact
package versions in [`scientific_code/requirements.txt`](scientific_code/requirements.txt).
These are version pins, not a platform-specific wheel lock. Other environments
have not been validated for this package. SageMath is not required.

Run the following from the repository root on macOS or Linux:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r scientific_code/requirements.txt
.venv/bin/python scientific_code/manifest.py --root .
.venv/bin/python scientific_code/manifest.py
```

On Windows, use the corresponding Python 3.14 interpreter and
`.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Checksum verification alone needs only Python 3.11 or later and the standard
library; it does not require installing the numerical dependencies.

## Run calculations

```sh
# List every available calculation without running it.
.venv/bin/python scientific_code/run.py --mode full --list

# Run the default selection of central calculations.
.venv/bin/python scientific_code/run.py

# Run selected calculations.
.venv/bin/python scientific_code/run.py --only quartic half_interval_scan
```

Fresh scientific results and logs go to `scientific_code/results/`, leaving the
supplied reference data unchanged. The runner records the environment, task
status, and elapsed time. Do not use `python -O` or `PYTHONOPTIMIZE`: scientific
checks rely on assertions. Some symbolic and high-precision tasks are expensive;
the complete suite was not rerun during packaging.

See [`scientific_code/docs/CLAIMS.md`](scientific_code/docs/CLAIMS.md) for the
scope of individual checks and
[`scientific_code/docs/REPRODUCING.md`](scientific_code/docs/REPRODUCING.md) for the
full task guide.

## Regenerate the PRL figures

```sh
.venv/bin/python prl_figures/plot_product_branches.py
.venv/bin/python prl_figures/plot_supplement_figures.py
.venv/bin/python scientific_code/run.py --only plot_deficit
```

The first two commands use supplied data to regenerate Letter Figure 1 and
Supplement Figures S2–S4 in `prl_figures/figures/`. The third writes Supplement
Figure S1 to `scientific_code/results/data/figures/deficit.pdf`.
Verify the distributed checksums **before** regenerating figures: the PRL plot
scripts overwrite their reference PDFs and may update metadata, so the supplied
manifest can subsequently report expected changes.

## Citation and license

Use [`CITATION.cff`](CITATION.cff) for this package's current title and authors.
The package identifier is `2026.11.03-prl5`; the scientific snapshot remains
`2026.11.02-r5`. This package is prepared for GitHub upload; no public release or
archival DOI is asserted.

License status is recorded in [`LICENSE_STATUS.md`](LICENSE_STATUS.md).
