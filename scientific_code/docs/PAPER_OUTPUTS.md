# Reproducing the paper's figures and tables

This map applies to version **2026.11.02-r5**, including the added unequal-slope results. Figure and table numbers refer to the accompanying paper;
LaTeX labels identify the results independently of pagination. [CLAIMS.md](CLAIMS.md)
also maps supplementary calculations and labels retained from earlier drafts.
Output paths below are relative to the selected data directory: `results/data/`
for new runs, or `data/generated/` for the supplied reference snapshot.

| Paper output | Data / scientific task | Plot or extraction task | Output |
| --- | --- | --- | --- |
| Figure 1 (`fig:branches`), one amplitude example and all five coupling-product curves on common axes | `branch_trace` (scaled autonomous shooting) | `plot_branches` | `branches/branch_trace.csv`, `figures/branches.pdf` |
| Figure 2 (`fig:deficit`), small-amplitude coefficient | `half_interval_scan` (independent radial half-interval shooting) | `plot_deficit` | `branch/amplitude_scan.csv`, `figures/deficit.pdf` |
| Figure 4 (`fig:orbit`), exterior mass and temperature | `orbit_scan` (fresh seed and compactified exterior integration) | `plot_orbit` | `orbit/orbit_data.csv`, `orbit/orbit_metadata.json`, `figures/orbit.pdf` |
| Figure 5 (`fig:families`), family-domain cartoon | Existing `orbit_scan` and `half_interval_scan` results | `plot_families` (schematic, no new physical solve) | `figures/families.pdf`, `figures/families_metadata.json` |
| Figure 3 (`fig:excess`), asymmetric scalar profile and entropy product | `asymmetric_excess` and fresh even-branch continuation | `plot_excess` | `asymmetric/profile_interior.csv`, `profile_exterior.csv`, `branch_r2_n2.csv`, `figures/excess_example.pdf` |
| Table 1 (`tab:windows`), sampled couplings and deficits | `branch_trace` | `numerical_regime` selects stated amplitudes and computes potential ratios and area bounds | `branches/numerical_regime.csv`, `.json` |
| Table 2 (`num:convergence`), independent radial refinements | `tuned_branch_bvp` and `test_tuned_branch_bvp` | Select the settings stated in the caption | `bvp/tuned_branch_bvp.json`, `bvp/independent_double_shoot.json` |
| Fixed-coupling point and endpoint refinements | `fixed_coupling_control` | No plot | `branches/fixed_coupling_control.json` |
| Stronger orbit gap bound and finite-amplitude regime | Existing `branch_trace` data | `numerical_regime` | `branches/numerical_regime.csv`, `.json` |

To regenerate all five figures from the supplied data and check the sampled
amplitude summary:

```sh
python3 run.py --only plot_branches plot_deficit plot_orbit plot_families numerical_regime plot_excess --log-dir results/validation/paper_outputs
```

The orbit plot recomputes its plotting samples from the exterior trajectory;
`plot_excess` also solves its asymmetric profile and branch afresh, while the other three plot tasks read existing data. To regenerate the
underlying scientific datasets first:

```sh
python3 run.py --only branch_trace fixed_coupling_control half_interval_scan orbit_scan tuned_branch_bvp test_tuned_branch_bvp
```

Add `--paper-figures-dir /path/to/paper/figures` to a plotting selection to copy
the successfully generated PDFs. Plot provenance files record input checksums.
A custom `--output-dir` is supported: the plotting/summary tasks
prefer fresh inputs there and otherwise explicitly record their packaged-data
fallback.

## Distinct method coverage

The amplitude-6 endpoints of Figure 1 are from scaled autonomous integration,
with outer-asymptote cutoff and integration-tolerance refinements. They are not
radial collocation solutions with a fixed radial horizon cutoff. Large horizon
potential ratios produce very narrow radial layers; the autonomous variables
avoid this loss of radial resolution.

The current independent overlap is:

- Full interval collocation, two-sided radial shooting and radial half-interval
  reflection shooting at outer scalar amplitude 0.2.
- Radial half-interval shooting versus scaled autonomous shooting at the seven
  common stored amplitudes 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35. The maximum
  coupling difference is below 1.6e-12 and product difference below 4.6e-13.
- A separate fixed-nu=2.99 radial half-interval solve checked against reflected
  inner-horizon integration. Its historical record is preserved as
  `branches/fixed_nu_2p99_radial_check.json`.

No independent radial method is claimed for the large-amplitude endpoints.
Refining the same autonomous solver, and checking it on an exact A2 solution,
are useful controls but do not create a second numerical method.

## Unequal-slope validation

The fixed theory in Figure 3 is checked by autonomous even matching,
two-sided radial shooting with orders 6, 8 and 10, and independent collocation.
The coupling identity uses the autonomous solutions and is a consistency check,
not a third independent construction in that script. Current task logs are in
`validation/excess/`; the collocation report is
`data/generated/asymmetric/independent_collocation.json`.

The asymmetric curves are finite amplitude traces, not a classification of all
solutions, a determination of endpoints, or a bound on eta at fixed slopes.
