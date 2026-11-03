# Paper-to-code map

For the current four figures and two tables, use [PAPER_OUTPUTS.md](PAPER_OUTPUTS.md). The broader
map below includes stable labels from earlier drafts and supplementary controls.


References below use the manuscript's stable LaTeX labels, rather than numbers
that change during typesetting. Output paths are relative to the selected data
directory (`results/data/` for new runs; `data/generated/` for supplied data).
Log names are relative to the selected log directory (`results/validation/MODE/`
for new runs; historical records are under `validation/reference/`). The table maps
computational claims and controls; general existence, integrability and global
topological statements remain analytic arguments in the paper.

Rows marked supplementary describe additional controls in the code package,
rather than derivations claimed in the paper. In particular, the general
angular, rotating-throat and complex-root calculations do not establish a
regular rotating hairy family or a global meromorphic continuation of a hairy
solution.

| Paper label / result | Implementation | Output or validation record |
| --- | --- | --- |
| `cps:iw`, `eq:closed`, `eq:K`, `cps:kinfinity`: surface-charge conventions and radial conservation | `scripts/cps.py`, `static_K.py`, `check_K_conservation.py`, `check_K_dyonic4d.py` | corresponding `.log` files; symbolic identities and explicit residuals |
| `cps:smooth`: regular representative at the bifurcation surface | `scripts/smooth_representative_check.py` | `smooth_representative_check.log`; smooth-vector and deliberately singular controls |
| `eq:radial`: reduced field equations | `scripts/reduced_equations_check.py`, `ddim_identity.py` | corresponding logs |
| `cold:factorization`, `cold:bracket`, `cold:countermodel`, `cold:floor`: entropy statements and limits | `scripts/cold_from_hot_check.py` | `cold_from_hot_check.log`; exact models and countermodels, not a replacement for the general proofs |
| `cold:static`, `cold:product`, `cold:smarr` | `scripts/dilaton_exact.py`, `dilaton_thermo.py`, `agm_checks.py` | corresponding logs; exact dilaton/Toda controls |
| `num:c2`, `num:c2exp`, `num:c3`, `num:beta`, `num:c3seven` | `scripts/c3_exact.py` | `c3_exact.log`; rational projections including the degree-seven cancellation |
| `cold:coupling-product`, `num:coupling-deficit`, `num:tuning-next`, `num:deficit-next` | `controls/coupling_series_control.py` | `coupling_series_control.log`; fixed-field coupling derivative with electric/magnetic/axionic terms, RN normalization and independent low-order scalar projection and entropy coefficients |
| `num:exactprojective`, `num:c5seven`, `num:c5nine` | `controls/exact_obstruction_control.py` | `controls/exact_obstruction_control.json`; exact rational source recurrence and first nonzero fifth order projection at the two exceptional couplings; excludes an analytic small-amplitude family at fixed coupling, not all finite-amplitude solutions |
| `num:todabtwo`, `num:todaendpoints` | `controls/toda_b2_control.py` | `controls/toda_b2_control.json`; exact B2 equations, constraint and endpoint identity, with independent IVP refinement; no claim that the symmetric BVP solver was tested at unequal couplings |
| `num:tuning-next`, `num:deficit-next`, `tab:amplitude-scan`: higher-order consistency comparison | `controls/series_fraction_comparison.py`, `fraction_polynomial.py` | `controls/series_comparison.csv`; powers 28/34 compared with unrounded `branch/amplitude_scan.csv`, with the six-row manuscript check at power 34 (`3e-13` in alpha, `1e-13` in eta); backend reproduction of the formal recurrence, not a convergence proof or an independent high-order derivation |
| `num:convergence` (convergence table), `num:result`, `num:endpoint`, `num:curvature`, Appendix `app:numseed` | `scripts/bvp_collocation.py`, `tuned_branch_bvp.py`, `test_tuned_branch_bvp.py` | `bvp/tuned_branch_bvp.json`, `bvp/independent_double_shoot.json`, `bvp/profiles/*.json` |
| `num:reversible-involution`, `num:reversible-shooting`, `tab:amplitude-scan` | `calculations/half_interval_scan.py` | `branch/amplitude_scan.csv`, `branch/half_interval_validation.json` |
| `num:windows`, `tab:windows`, `num:fixed-nu-example`: fixed-coupling matching, five finite-amplitude windows and scalar node labels | `calculations/reversible_branches.py`, `branch_trace.py`; `controls/fixed_coupling_control.py` | `branches/branch_trace.csv`, `branches/branch_trace.json`, `branches/fixed_coupling_scan.csv`, `branches/fixed_coupling_control.json`; 45 amplitudes per branch through 6, fixed-theory solves, three-setting endpoint refinements and A2 controls |
| `fig:branches`: coupling and entropy-product branch figure | `calculations/plot_branches.py` | `figures/branches.pdf`, `figures/branches.png`, `branches/figure_metadata.json`; source CSV checksum, analytic zero-amplitude markers and finite sampled endpoints distinguished |
| `app:higher-modes`, `intro:higher-modes`: exact quartic signs for modes 1--20 | `calculations/higher_modes.py` | `higher_modes/higher_modes_results.tsv`, `higher_modes/provenance.json`; independently validated reference results, not freshly rerun in the 2 November selection; standalone/runner task available |
| `num:quartic-deficit`, `num:quartic-reduced`, `num:quartic-solvability`, `intro:quartic` | `calculations/quartic.py`; independent numerical comparison in `half_interval_scan.py` | `branch/quartic_exact.json`, `branch/amplitude_scan.csv` |
| Central numerical reference-value regressions | `calculations/validate_core.py` | `branch/core_validation.json`; expected-number guards, not independent derivations |
| `eq:full-orbit-action`, `num:af-representatives`, `num:af-target-map`, `num:af-map-identities`, `num:af-collar-asymptotics`, `num:af-fixed-charges` | `controls/general_translation_check.py` | `controls/general_translation.json`; Jacobian, radial constraint, force and fixed-charge identities; regular exterior collar and asymptotic hypotheses are analytic arguments in the paper |
| Supplementary singular-endpoint flux identities | `controls/general_translation_check.py` | `controls/general_translation.json`; exact magnetic-dilaton example with boundary fluxes retained; not a classification of generic hairy interiors |
| `eq:full-orbit-action`, `eq:orbit-physical-data`, `tab:orbit-physical-data` | `calculations/orbit_scan.py` | `orbit/orbit_data.csv`, `orbit/orbit_metadata.json`; includes entropy gap and physical observables |
| `eq:orbit-critical-radius`, `eq:orbit-gap-infimum`, `eq:orbit-runaway-scaling`, `eq:orbit-large-radius`, `tab:orbit-landmarks`, `app:compact-system` | `calculations/orbit_scan.py` | `orbit/orbit_metadata.json`, `orbit/orbit_convergence.json`; critical radius, limiting gap, mass minimum, temperature maximum and endpoint exponents |
| `fig:orbit` | `calculations/plot_orbit.py` | `figures/orbit.pdf`, `figures/orbit.png`, `orbit/orbit_plot_data.csv`; logarithmic horizontal and vertical scales, lower relative-radius sample `1e-10`; optional `--paper-figures-dir` copies the fresh PDF after a successful plot task |
| `cont:map`, `cont:dualmap`, `cont:balance`, `cont:exactflux`, `cont:entropyratio`, `cont:flatmetric`, `cont:flatflux` | `scripts/klr_continuation.py`, `continued_bvp.py` | corresponding logs; `continuation/continued_bvp.json` |
| `cont:entropy-hessian`, `avir:reduced-maxwell`, `avir:reduced-entropy`, `avir:entropy-second-variation` | `controls/second_variation_control.py` | `controls/second_variation.json`; exact spherical Legendre Hessian, fresh collocation profiles, quadrature/endpoint-order refinement, charge-preserving entropy finite differences and stationary first variations |
| `cold:average_scalar`, `cold:average_metric`, `cold:secondmetric`, `cold:sphericalfunctional`, `virial:routhian`, `avir:mapped-metric-average`, `avir:mapped-scalar-average` | `controls/check_static_virial.py` | `check_static_virial.log`, `controls/check_static_virial.log` |
| Supplementary extremal Kerr and angular controls | `controls/rotating_exact_controls.py` | `rotating_exact_controls.log`; exact angular metric equations, axis contribution and spherical controls |
| `ext:rod`: rotating products and first laws | `scripts/kerr_newman_check.py`, `kerr_sen_check.py`, `region2_rotating.py` | corresponding logs; electrovacuum/axion-dilaton examples, not a generic rotating hairy branch |
| Supplementary rotating throat-limit checks | `scripts/rotating_attractor_limit.py` | `rotating_attractor_limit.log`; explicit rotating controls, not a general rotating extension of the tuned branch |
| Supplementary RN–AdS and polynomial residue identities | `scripts/complex_horizons.py`, `complex_horizons_symbolic.py`; `controls/complex_validation.py` | corresponding logs and `controls/complex_validation.json`; exact and numerical complex-root controls |
| Supplementary exact geometries | `scripts/toda_exact.py`, `subtracted_rn.py`, `btz_check.py` | corresponding logs |
| Inward integration obstructions and supplementary logarithmic-mode figure | `scripts/inner_regularity_physical.py`, `fig_inner_regularity.py` | corresponding full-mode logs, `figures/inner_regularity.*`; these do not replace the two-horizon BVP evidence |
| Negative controls, finite samples, valid domains and exact numerical benchmarks | `scripts/test_checklib_regressions.py`, `test_inner_regularity_validation.py`, `test_exact_dilaton_integration.py`, `test_eom_num.py`, `kerr_sen_eom_grid.py`, `kerr_sen_fields.py` | corresponding logs |

The implicit-function existence theorem is not established by a numerical test.
The finite-amplitude construction has empirical convergence evidence; the
matching curve's local analytic existence is proved separately. Complex-root
controls do not establish a global meromorphic continuation for hairy solutions.

The five branch traces concern the reflection-symmetric cosh model only.
Their lowest sampled couplings at amplitude 6 are not certified global window
edges, and sampled deficits are not certified maxima. Discrete monotonicity
checks do not prove a continuous global sign. A finite scan without a detected
sign-changing bracket does not exclude tangent roots, closely spaced pairs,
larger amplitudes, or nonreversible solutions. Each accepted traced point has
its full scalar node count checked independently of the branch label. No
generic asymmetric-coupling continuation or higher-mode overlap scan is claimed.

The exact-coefficient controls distinguish computed source projections and
field-equation residuals from assertions against known constants. The latter
are regression guards. An endpoint identity at asymmetric Toda couplings does
not imply scalar sign reflection of the full field equations.

`calculations/numerical_regime.py` summarizes the accepted branch data for selected
amplitudes, horizon-area hierarchy and sharper gap bounds, with outputs
`branches/numerical_regime.csv` and `.json`. The script also compares stored
radial and autonomous data over their actual overlap; no new ODE solve is
implied. `plot_deficit.py` and `plot_families.py` reproduce the deficit plot and
family-domain cartoon.

## Unequal slopes (r4)

- `num:linear-crossing` and `num:cubic-excess` are analytic local projections:
  unequal slopes produce a nonzero linear crossing at positive even modes.
- `asymmetric_excess` checks `num:excess-example` using two shooting formulations
  and the asymptotically flat exterior. The coupling integral is a consistency check.
- `asymmetric_collocation` independently reproduces eta within 2e-12 from
  analytic RN-plus-P2 seed data, with unused Hamiltonian checks.
- `asymmetric_branch_check` samples deficits and excesses at several ratios,
  including B2. It does not prove a global boundary or uniqueness.
- `plot_excess` produces Figure 3 with sampled endpoints at phi_plus=+/-3.
- `cold:eta-upper` is an analytic pointwise bound. With fixed horizon amplitude
  it gives a uniform positive orbit gap. It does not bound eta by slopes alone.

## Manuscript revision (r5)

The local relation `num:local-excess-coupling` follows by eliminating the
amplitude between `num:linear-crossing` and `num:cubic-excess`:
eta - 1 = 2 (nu - n)^3 / (3 kappa_n^2) + O((nu - n)^4).
It describes the constructed local branch at fixed unequal slope ratio,
not an optimization over all regular branches. Figure 1 now uses common axes
for all five symmetric eta-versus-nu curves, with the same reference samples.
