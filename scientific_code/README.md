# Hot attractors and entropy bounds

Numerical and symbolic calculations accompanying the paper by Kevin Goldstein,
Vishnu Jejjala and Suresh Nampuri. The code computes two-horizon black-hole
solutions, entropy-product deficits and excesses, fixed-coupling branches, exterior orbits
and supporting exact controls.

This repository contains code, reference data and validation records; the paper
is distributed separately. Version **2026.11.02-r5** combines the five entropy-versus-coupling curves
on common axes beneath one amplitude example. It retains the unequal-slope
results and independent validation from r4; the numerical data are unchanged.

## Installation

The recorded scientific environment uses Python **3.14.3** and the exact package
versions in `requirements.txt`. No Sage installation is required.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python run.py
```

On Windows, use `.venv\Scripts\python.exe` in place of `.venv/bin/python`.
Use that interpreter for the `python3` commands below. Other Python versions and
platforms have not been validated as part of this release. Run without `-O` or
`PYTHONOPTIMIZE`: assertions are part of the scientific checks.

## Run the calculations

```sh
python3 run.py                         # central results (quick mode)
python3 run.py --mode standard         # additional exact and regression checks
python3 run.py --mode full --list      # inspect every available task
python3 run.py --only quartic          # one exact calculation
python3 -m unittest discover -s tests -v  # runner and output-routing regressions
```

New data go to `results/data/`, and logs, timings, environment details and
pass/fail summaries go to `results/validation/`. These directories are ignored
by Git. The supplied `data/generated/` and `validation/reference/` records are
kept separate from new runs. The runner works from any working directory.

To regenerate the paper's five figures and amplitude summary from the supplied
data:

```sh
python3 run.py --only plot_branches plot_deficit plot_orbit plot_families numerical_regime plot_excess
```

Plots appear in `results/data/figures/`. Input readers prefer newly generated
data in the selected output directory and otherwise use the supplied reference
data. The orbit plot recomputes its plotting trajectory. See the
[figure and table map](docs/PAPER_OUTPUTS.md) for the underlying calculations.

Quick mode covers the central numerical and exact calculations. Standard mode
adds controls and the powers-28/34 series comparison. Full mode includes costly
symbolic tensor and high-precision calculations and can take hours; the complete
current full suite has not been timed or rerun. Use `--list` to see the selection
and `--timeout SECONDS` for an optional per-task limit. Any failing task makes
the runner return a nonzero exit status.

## Repository guide

| Path | Contents |
| --- | --- |
| `run.py` | Task selection, output routing and execution records |
| `calculations/` | Entropy expansion, branch shooting, exterior orbit and figures |
| `scripts/` | Collocation, two-sided shooting, covariant charges and exact controls |
| `controls/` | Coupling/series calculations, Toda checks, translation identities and Hessian checks |
| `tests/` | Runner regressions |
| `data/generated/` | Supplied scientific reference data and figures |
| `validation/reference/` | Historical task logs and environment records |
| `docs/` | Reproduction instructions, paper-to-code maps and validation coverage |
| `manifest.py`, `MANIFEST.sha256` | Integrity check for the distributed snapshot |

- [Detailed reproduction instructions](docs/REPRODUCING.md)
- [Figures and tables](docs/PAPER_OUTPUTS.md)
- [Scientific claims and controls](docs/CLAIMS.md)
- [Validation coverage and measured timings](docs/VALIDATION.md)

The five deficit windows concern the symmetric exponential-coupling model.
Additional unequal-slope controls establish a regular excess example and
explore both signs over stated finite ranges. Their endpoints and refinement spreads are numerical evidence,
not certified global bounds. Supplementary rotating and complex-root controls
do not establish corresponding generic hairy solution families. The detailed
maps distinguish independent methods from regression guards.

## Citation and license

Citation metadata are in [CITATION.cff](CITATION.cff), and snapshot details are in
[RELEASE.json](RELEASE.json). No archival DOI has been assigned. The authors have
not selected a license; [LICENSE_STATUS.md](LICENSE_STATUS.md) records that status.

To verify the supplied files, run `python3 manifest.py`. After intentionally
changing release files, regenerate the manifest with `python3 manifest.py --write`.
Normal runs under `results/` do not change the reference snapshot.

## Unequal-slope excess example

```sh
python3 run.py --only asymmetric_excess asymmetric_collocation asymmetric_branch_check plot_excess
```

`asymmetric_excess` solves the fixed theory with slopes `(sqrt(7),-2*sqrt(7))`
and squared charges `(2,1)`, using autonomous and two-sided radial shooting.
It also integrates the coupling identity as a consistency check and constructs
an asymptotically flat exterior. `asymmetric_collocation` uses an analytic
RN-plus-Legendre seed and a separate collocation formulation. They give
`eta = 1.02205318802`. `asymmetric_branch_check` samples additional ratios and
includes the exact B2 control. `plot_excess` generates its profile, both signed
amplitude traces and the figure, with fresh scientific solves.

These tasks are in standard/full modes; the 22-task quick selection is retained.
The maximum eta at fixed slopes is not established. The ~21% excess is a sample
on a varying-coupling curve. The positive fixed-amplitude orbit gap bound is
not asserted uniformly over families in which the amplitude changes.
