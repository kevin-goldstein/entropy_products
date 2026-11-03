# Validation records and scope

Version **2026.11.02-r3** separates supplied reference records from new runs.
The historical scientific records below come from **2026.11.02-r2** and its
predecessors. Dates and task selections are preserved; they are not relabeled
as fresh executions. The complete current full suite has not been run.

The recorded scientific environment is CPython 3.14.3 on macOS 27.0.1 arm64 with
NumPy 2.4.4, SciPy 1.17.1, SymPy 1.14.0, mpmath 1.4.1 and Matplotlib 3.10.8.
Each runner record includes its own machine-readable environment. No Sage API
or python-flint dependency is required.

Paths in this document are relative to the repository root. Supplied records
live in `validation/reference/`; new runs write to `results/validation/`.
Historical machine-specific paths have been normalized for distribution; see
[validation/README.md](../validation/README.md). Scientific values, task outcomes and timings
have been retained.

## Repository packaging checks (2 November 2026)

For version **2026.11.02-r3**, all **22 current quick-mode tasks passed** in a
fresh copy with a path containing spaces, invoked from outside the repository.
The run took **102.749 seconds** in the recorded environment. All **12 runner
and output-path regression tests passed**. An additional seven-task selection
checked core calculations and all four plots using packaged-input fallbacks.

The reference-file manifest verified unchanged after the quick run. Reference
numerical values, tables, profiles and figure files are preserved from r2;
metadata path strings were normalized. Details and task-level timings are in
[validation/packaging.json](../validation/packaging.json). These checks used the
installed pinned dependencies; they do not claim a fresh package-index install
or a run of full mode.

## Historical selections

| Date | Selection | Result | Wall time | Reference record |
| --- | --- | --- | --- | --- |
| 2 Nov 2026 | Branch trace, fixed-coupling control and branch figure | 3 tasks passed | 64.818 s | `validation/reference/fixed_coupling/summary.json` |
| 2 Nov 2026 | Runner regressions | 8 tests passed | See log | `validation/reference/fixed_coupling_runner/test_runner.log` |
| 2 Nov 2026 | Four figures and numerical-regime summary | 5 tasks passed | 19.604 s | `validation/reference/paper_outputs/summary.json` |
| 2 Nov 2026 | Final deficit inset layout | 1 task passed | See summary | `validation/reference/deficit_layout/summary.json` |
| 25 Oct 2026 | Powers 28/34 against supplied scan | 1 task passed | 46.643 s | `validation/reference/series_packaged/summary.json` |
| 25 Oct 2026 | Fresh half-interval scan and powers 28/34 | 2 tasks passed | 48.535 s | `validation/reference/series_regenerated/summary.json` |
| 25 Oct 2026 | Coupling/series, obstruction, B2 and translation controls | 5 tasks passed | 55.883 s | `validation/reference/analytic_controls/summary.json` |
| 25 Oct 2026 | Series input-routing check | 1 task passed | 46.015 s | `validation/reference/series_input_routing/summary.json` |
| 25 Oct 2026 | Orbit figure and optional export | 1 task passed | 15.026 s | `validation/reference/figure_export/summary.json` |
| 25 Oct 2026 | Runner failure handling and figure copying | 8 tests passed | 0.313 s | `validation/reference/runner_analytic_controls/test_runner.log` |
| 24 Oct 2026 | Quick mode at that revision | 12 tasks passed | 36.046 s | `validation/reference/quick/summary.json` |
| 24 Oct 2026 | Standard mode at that revision | 33 tasks passed | 145.443 s | `validation/reference/standard/summary.json` |
| 24 Oct 2026 | Runner failure handling | 6 tests passed | 0.268 s | `validation/reference/runner/test_runner.log` |
| 24 Oct 2026 | Core regression guards | 1 task passed | 1.130 s | `validation/reference/selected/summary.json` |

All listed selections had zero failures. Quick and standard task lists have
expanded since 24 October; those historical records alone do not establish a pass
for the current modes; the current quick-mode run is recorded above. Historical full-task listings describe the selection
available on their recorded dates, not completed executions. Use
`python3 run.py --mode full --list` for the current list.

Timings include process startup and, for plotting, a fresh temporary Matplotlib
cache. They are measurements, not performance guarantees. Selections overlap
and must not be added to count independent scientific validations. PASS-marker
totals exclude uncounted assertions. The summaries contain per-task exit codes,
timings and marker counts.

## Fixed-coupling branches

The trace contains 225 samples: 45 positive amplitudes from 0.05 through 6 for
each node number 3–7. Every point has the intended full scalar node count and a
strict numerical entropy-product deficit. The largest absolute energy residual
is `5.15e-12`; the largest amplitude-normalized center matching residual is
`8.50e-11`. Consecutive sampled nu and eta values decrease in the retained run.
This discrete trend is not a global monotonicity proof; roundoff-scale reversals
can arise on nearly flat tails with looser root tolerances.

Three settings, `(T, rtol) = (16,1e-11), (18,2e-12), (20,3e-14)`, were compared
at amplitude 6. The largest spreads are `3.44e-12` in nu and `1.89e-9` in eta.
These are empirical spreads, not rigorous error bounds. Rounded finest-setting
results are:

| Scalar nodes | nu at phi_plus=6 | Deficit 1-eta |
| --- | --- | --- |
| 3 | 2.950631 | 3.6330% |
| 4 | 3.881127 | 6.3207% |
| 5 | 4.801548 | 8.1676% |
| 6 | 5.716216 | 9.4633% |
| 7 | 6.627292 | 10.4021% |

At fixed nu=2.99, the finest solve gives `phi_plus=0.34680368180891` and
`eta=0.998934679598087`. The amplitude-0.2 reference point and inverse
fixed-coupling solve agree with the separate radial shooting/collocation
implementations. Exact A2 controls at amplitudes 0.2, 1, 3 and 6 have maximum
`abs(eta-1)` of `1.07e-13`.

For each nu in `2.5, 2.9, 2.99, 3.01, 3.5`, 128 positive amplitudes in
`[0.005,6]` were sampled for each center parity condition. Only the nu=2.99 odd
condition has a detected sign-changing root, with three scalar nodes. Absence
of a detected bracket does not exclude tangent roots, close pairs, larger
amplitudes or nonreversible solutions. These finite samples do not certify
limiting couplings or maximum deficits. That historical selection did not scan unequal exponential slopes or
higher-mode window overlap; the r4 unequal-slope extension is documented below.

The higher-mode exact script and twenty-row rational table are retained from
earlier independent checks of the cubic projection and quartic metric response
for n=1,...,20. Provenance is in `data/generated/higher_modes/provenance.json`.
No fresh symbolic execution is claimed by the 2 November selection. The
supplementary `higher_modes` task is available explicitly and in full mode.

## Series comparison

The two 25 October series runs used the supplied scan and a fresh half-interval
scan, respectively. The regenerated scan was byte-for-byte identical to the
supplied one. At the six manuscript-table amplitudes, both power-34 comparisons
have maximum absolute differences `2.042e-13` in alpha and `6.029e-14` in eta,
below the empirical assertions `3e-13` and `1e-13`. These repeat the same
recurrence and shooting method; they are not additional independent numerical
constructions.

At amplitude 0.3, power 34 minus power 28 is `+1.440e-13` in alpha and
`-1.047e-13` in eta. Power 28 partly cancels the shooting error: its eta
difference is `+4.441e-14`, compared with `-6.029e-14` at power 34. At amplitudes
at or below 0.2, the power changes are below `1e-18`. Power 28 remains a
truncation diagnostic; the agreement assertion uses power 34. The extra
amplitude 0.35 is reported separately and is not covered by the six-row maxima.

A separately supplied Linux x86-64 calculation crossed the former power-28
threshold but met the power-34 thresholds. That environment was not rerun as
part of this package. Floating-point ODE and root-solver results can vary across
platforms. These tolerances measure agreement with the half-interval solver,
not certified accuracy, and do not override larger observed differences among
independent constructions. No rigorous remainder bound or convergence radius
is inferred. Raw comparisons and checksums are retained in
`validation/reference/series_comparison_summary.json`.

To repeat both selections in new output directories:

```sh
python3 run.py --only series_fraction_comparison --log-dir results/validation/series_packaged --output-dir results/series_packaged
python3 run.py --only half_interval_scan series_fraction_comparison --log-dir results/validation/series_regenerated --output-dir results/series_regenerated
```

## Numerical and exact controls

Selected quantities reproduced on 24 October were:

| Quantity | Value |
| --- | --- |
| Collocation coupling | 4.893760362084041 |
| Collocation normalized entropy product | 0.9998599664739966 |
| Half-interval coupling | 4.893760362084386 |
| Half-interval normalized entropy product | 0.9998599664742153 |
| Exact quartic deficit coefficient | 96/1001 |
| Exterior critical outer radius | 4.203969175802427 |
| Exterior gap infimum in units of S_ext | 8.723529569403414 |
| Minimum normalized mass | 3.409746303411836 |
| Maximum normalized outer temperature | 0.013217923028129451 |

The next coefficients are `d4 = -38719078400/323969969323` in `3-nu` and
`eta6 = 51807744/231175945`. The SymPy control obtains the entropy coefficient
from the coupling law; the portable formal recurrence evaluates powers 28 and
34. Agreement between arithmetic implementations is not an independent
high-order derivation or a convergence proof.

At exceptional indices 7 and 9, exact projection yields vanishing quadratic,
cubic and quartic obstructions and a nonzero fifth-order coefficient. This
concerns analytic families through the constant-scalar solution at fixed
coupling, not global nonexistence. The exact B2 control checks all reduced
equations, the constraint and endpoint identity, and compares a separate IVP
at two tolerances; it does not validate the symmetric BVP implementation at
unequal couplings.

The translation control checks the map, constraint, force weights and charges,
including axionic mixing, and a singular magnetic-dilaton endpoint example.
These identities do not replace the analytic regular-collar argument or
classify generic singular interiors. The entropy Hessian control uses fresh
BVP profiles, endpoint-order and quadrature refinements, charge-preserving
finite differences and stationary first variations. The opposite P1/P4 signs
identify a saddle of the static functional, not Lorentzian dynamical stability.

The orbit's smallest plotting sample, `(B-Bcrit)/Bcrit=1e-10`, was compared
between outer launch cutoffs `1e-6` and `1e-7`; relative changes were `1.27e-6`
in mass and `2.12e-7` in temperature. See
`validation/reference/runner/orbit_plot_tail_check.json`.

The numerical-regime summary derives area hierarchies and stronger gap bounds
from accepted data and compares the actual radial/autonomous overlap; it does
not solve new interior ODEs. The orbit plot recomputes its trajectory from a
fresh seed. The other figure tasks read data. All four figures were rendered
and inspected in the 2 November snapshot; the family-domain figure is a
schematic, not an independent solution scan.

## Runner and integrity checks

Runner regressions use disposable programs to test nonzero exits, active
assertions, timeouts, `[FAIL]` markers, output routing, task listing and figure
copying. Run `python3 -m unittest discover -s tests -v` for the current runner
and output-path tests. The checksum utility was tested on
disposable files: unchanged files verified, while changed, missing and unlisted
files failed. Historical results are in
`validation/reference/runner/manifest_checks.json`.

Run `python3 manifest.py` to verify the distributed snapshot. New calculations
under ignored `results/` do not change it. After deliberately updating release
sources, reference data or metadata, regenerate with `python3 manifest.py --write`
and verify again before distributing a new snapshot.

See [CLAIMS.md](CLAIMS.md) for the distinction among analytic arguments,
independent numerical methods and regression guards, and
[PAPER_OUTPUTS.md](PAPER_OUTPUTS.md) for method coverage of each figure and table.

## Unequal-slope extension (r4)

Five selected tasks passed in 49.599 s: asymmetric_excess,
asymmetric_collocation, asymmetric_branch_check, plot_branches and plot_excess.
The branch figure was rerun after a final layout adjustment, and all 12
runner/path tests passed. Logs and exact task selections are in
`validation/excess/`. This selection is not a rerun of standard/full mode.
The collocation result has eta 1.0220531880176, two scalar nodes and
max |b^2 H| below 2e-11; shooting gives 1.0220531880189.

## Combined branch figure (r5)

The branch plot was regenerated as three panels: the n=3 amplitude example
above all five eta-versus-nu curves on common axes. All 225 reference samples
are byte-identical to r4. The rendered figure was visually checked; the log
and source checksum are recorded in `validation/review/`. No new numerical
solutions are claimed by this plotting revision.
