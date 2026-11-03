r"""Reproducible independent two-horizon validation of the tuned branch.

Run: python3 tuned_branch_bvp.py [--output-dir DIRECTORY]
Outputs default to results/data/bvp in the package; every setting has its own profile file.
The main convergence study starts every solve afresh from RN + P_3 data.
For the same-theory study, alpha is held at the converged baseline value and
phi+ is an unknown. An auxiliary alpha-tuned collocation at the new b+ supplies
only the initial guess, and its diagnostics are retained separately. No inward
integration or inward-shooting data seed any of these solves.

All errors and spreads are empirical numerical diagnostics, not rigorous
interval enclosures or a proof of global branch existence. The exterior check
only integrates outward AFTER the two-horizon collocation has converged.
"""
import argparse
import json
from pathlib import Path
import os
import platform
import time
import numpy as np
import scipy

import checklib as ck
from bvp_collocation import (BDE, DELTA, RP, RM, BVPError, solve_branch,
                             exterior_check, taylor_coefficients)


SETTINGS = [
    ('order4_coarse', dict(order=4, cutoff=3e-5, tolerance=1e-8, initial_nodes=151)),
    ('order6_medium', dict(order=6, cutoff=1e-4, tolerance=1e-9, initial_nodes=301)),
    ('order6_fine', dict(order=6, cutoff=1e-4, tolerance=1e-10, initial_nodes=401)),
    ('order8_larger_cutoff', dict(order=8, cutoff=3e-4, tolerance=1e-10, initial_nodes=151)),
    ('order8_fine', dict(order=8, cutoff=1e-4, tolerance=1e-10, initial_nodes=601)),
]


def scaled_difference(actual, reference):
    return (np.asarray(actual)-reference)/np.maximum(1, np.abs(reference))


def profile_record(solution, points=501):
    """Fields and on-shell invariant profiles, including analytic horizon limits."""
    z = np.linspace(-1, 1, points)
    r = solution.radius(z)
    r[0], r[-1] = RM, RP
    b, bd, phi, pd = solution.sample(r)
    p = (r-RP)*(r-RM)
    dp = 2*r-RP-RM
    bdd = -b*pd**2
    pdd = np.empty_like(r)
    pdd[1:-1] = (solution.alpha*np.sinh(solution.alpha*phi[1:-1])/b[1:-1]**2-dp[1:-1]*pd[1:-1])/p[1:-1]
    for index, args in ((0, (solution.bm, solution.bdm, solution.phim, -DELTA)),
                        (-1, (solution.bp, solution.bdp, solution.phip, DELTA))):
        _, coefficients = taylor_coefficients(*args[:3], solution.alpha, args[3], solution.order)
        pdd[index] = 2*coefficients[2]
    aa = p/b**2
    ad = dp/b**2-2*p*bd/b**3
    add = 2/b**2-4*dp*bd/b**3-2*p*bdd/b**3+6*p*bd**2/b**4
    ricci = -add-4*(aa*bdd+ad*bd)/b+2*(1-aa*bd**2)/b**2
    kretschmann = add**2+2*(ad*bd/b)**2+2*((2*aa*bdd+ad*bd)/b)**2+4*(1-aa*bd**2)**2/b**4
    h = -1+dp*bd/b-p*bd**2/b**2+2*np.cosh(solution.alpha*phi)/b**2-p*pd**2
    matrix = np.asarray([r, b, bd, bdd, phi, pd, pdd, aa, ricci, kretschmann, h]).T
    if not np.all(np.isfinite(matrix)):
        raise BVPError('non-finite saved profile')
    return dict(columns=['r', 'b', 'b_prime', 'b_second', 'phi', 'phi_prime', 'phi_second',
                         'a_squared', 'Ricci_scalar', 'Kretschmann', 'Hamiltonian'],
                values=matrix.tolist(),
                note='Second derivatives and invariants use the evolved equations; independent collocation-derivative residuals are in the summary. Horizon values use regular Taylor limits.')


def checked_diagnostics(label, solution, strict=False):
    d = solution.diagnostics()
    ck.true(label+': successful complete collocation', d['success'] and d['status'] == 0)
    ck.small(label+': requested relative collocation residual', d['collocation_max_rms'], solution.tolerance*1.01)
    ck.small(label+': both Taylor boundary matches', d['boundary_max_abs'], 1e-10)
    ck.small(label+': ODE residual at independent off-mesh points', d['off_grid_scaled_ode_max'], 1e-8 if strict else 1e-6)
    ck.small(label+': Hamiltonian first integral b²H', d['hamiltonian_constant_max_abs'], 1e-9 if strict else 1e-7)
    ck.small_all(label+': both horizon Hamiltonian limits', d['horizon_hamiltonian'], 1e-8)
    ck.small(label+': independently differentiated Ricci equation', d['ricci_scaled_equation_error'], 2e-8 if strict else 2e-6)
    ck.true(label+': positive b and finite horizon curvature',
            d['minimum_b'] > 0 and np.all(np.isfinite(d['kretschmann_horizons'])))
    r = solution.radius(np.linspace(-.999, .999, 501))
    scalar = solution.sample(r)[2]
    nodes = int(np.sum(np.signbit(scalar[1:]) != np.signbit(scalar[:-1])))
    ck.true(label+': nontrivial three-node branch retained', nodes == 3 and abs(solution.phip) > .19)
    d['interior_scalar_nodes'] = nodes
    return d


class ExactDilaton:
    """Independent GJN2014 Appendix A.1 closed form, alpha=2, Q1=Q2=1."""
    def __init__(self, mass, phi_inf):
        self.mass, self.phi_inf = mass, phi_inf
        self.sigma = -np.sinh(2*phi_inf)/mass
        self.delta = np.sqrt(mass**2+self.sigma**2-2*np.cosh(2*phi_inf))
        self.scale = 2*self.delta/DELTA

    def fields(self, r):
        radius = self.mass+self.scale*(np.asarray(r)-(RP+RM)/2)
        b = np.sqrt(radius**2-self.sigma**2)
        return np.asarray([b, self.scale*radius/b, self.phi_inf+np.arctanh(self.sigma/radius),
                           -self.scale*self.sigma/b**2])

    def kretschmann(self, r):
        b, bd, _, _ = self.fields(r)
        bdd = -self.scale**2*self.sigma**2/b**3
        p = (r-RP)*(r-RM)
        dp = 2*r-RP-RM
        aa = p/b**2
        ad = dp/b**2-2*p*bd/b**3
        add = 2/b**2-4*dp*bd/b**3-2*p*bdd/b**3+6*p*bd**2/b**4
        return add**2+2*(ad*bd/b)**2+2*((2*aa*bdd+ad*bd)/b)**2+4*(1-aa*bd**2)**2/b**4


def exact_benchmark(mass, phi_inf, directory):
    exact = ExactDilaton(mass, phi_inf)
    bp, _, phip, _ = exact.fields(RP)
    solution = solve_branch(bp=bp, amplitude=phip, degree=1, initial_nodes=401)
    d = solution.diagnostics()
    label = 'exact alpha=2 M=%g phi_inf=%g' % (mass, phi_inf)
    ck.true(label+': collocation succeeds', d['success'])
    ck.close(label+': recovered coupling', solution.alpha, 2., 1e-9)
    r = solution.radius(np.linspace(-1, 1, 801))
    r[0], r[-1] = RM, RP
    actual, reference = solution.sample(r), exact.fields(r)
    errors = []
    for i, field in enumerate(('b', 'bprime', 'phi', 'phiprime')):
        error = float(np.max(np.abs(scaled_difference(actual[i], reference[i]))))
        ck.small(label+': full profile including both horizons '+field, error, 2e-9)
        errors.append(error)
    profile = profile_record(solution, 801)
    curvature = np.asarray(profile['values'])[:, 9]
    ck.small_all(label+': full curvature profile', scaled_difference(curvature, exact.kretschmann(r)), 5e-9)
    ck.close(label+': unit area product', solution.product, 1., 1e-9)
    ck.small(label+': Hamiltonian first integral', d['hamiltonian_constant_max_abs'], 1e-9)
    ck.small(label+': independent off-mesh ODE residual', d['off_grid_scaled_ode_max'], 1e-8)
    # A wrong scalar-charge sign is readily distinguished by this oracle.
    ck.large_any(label+': rejects reversed scalar charge', actual[2]-(2*phi_inf-reference[2]), 1e-3)
    name = 'exact_M%s_phi%s.json' % (mass, phi_inf)
    write_json(directory/'profiles'/name, profile)
    d.update(mass=mass, phi_infinity=phi_inf, profile_errors=errors, profile_file='profiles/'+name)
    return d


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def run(output_dir):
    started = time.perf_counter()
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    solutions, rows = [], []
    for label, settings in SETTINGS:
        solution = solve_branch(**settings)
        d = checked_diagnostics(label, solution, strict=settings['tolerance'] <= 1e-10)
        d['label'], d['seed'] = label, 'analytic RN plus Legendre P3; fresh solve'
        d['profile_file'] = 'profiles/'+label+'.json'
        write_json(directory/d['profile_file'], profile_record(solution))
        rows.append(d)
        solutions.append(solution)
    reference = solutions[-1]
    radii = reference.radius(np.linspace(-1, 1, 1001))
    radii[0], radii[-1] = RM, RP
    reference_fields = reference.sample(radii)
    for solution, d in zip(solutions, rows):
        error = np.max(np.abs(scaled_difference(solution.sample(radii), reference_fields)), axis=1)
        d['profile_difference_from_finest'] = error.tolist()
        ck.small(d['label']+': alpha convergence', solution.alpha-reference.alpha, 1e-8)
        ck.small(d['label']+': area-product convergence', solution.product-reference.product, 1e-8)
        ck.small_all(d['label']+': all four fields converge including horizon derivatives', error, 1e-7)
        curvature_difference = scaled_difference(d['kretschmann_horizons'], rows[-1]['kretschmann_horizons'])
        d['horizon_curvature_difference_from_finest'] = np.asarray(curvature_difference).tolist()
        ck.small_all(d['label']+': horizon curvature convergence', curvature_difference, 1e-7)
    finest = rows[-3:]
    alpha_spread = float(np.ptp([r['alpha'] for r in finest]))
    product_spread = float(np.ptp([r['normalized_product'] for r in finest]))
    ck.small('three strict settings: coupling spread', alpha_spread, 2e-10)
    ck.small('three strict settings: product spread', product_spread, 2e-10)
    ck.large('resolved sub-attractor area-product deficit', 1-reference.product, 1.3e-4)
    ck.true('deficit is much larger than empirical convergence spread',
            (1-reference.product) > 1e4*max(product_spread, 1e-12))
    ck.close('regression: root agrees with reference inward result',
             reference.alpha, 4.893760362138778, 1e-8)

    fixed_rows, preparation_rows = [], []
    for kb in (2.5, 3., 4.):
        if kb == 2.5:
            # Correct the baseline at fixed alpha too; it is not merely relabelled.
            seed = reference
        else:
            seed = solve_branch(kb=kb, initial_nodes=401)
            sd = seed.diagnostics()
            sd.update(label='fixed_alpha_seed_kb%g' % kb,
                      seed='fresh analytic RN plus Legendre P3; alpha free only in seed preparation')
            preparation_rows.append(sd)
        solution = solve_branch(kb=kb, alpha=reference.alpha, seed_solution=seed, initial_nodes=401)
        label = 'fixed_alpha_kb%g' % kb
        d = checked_diagnostics(label, solution, strict=True)
        d.update(label=label, kb=kb, seed='auxiliary BVP at the same b+, followed by an exactly fixed-alpha correction',
                 profile_file='profiles/'+label+'.json')
        ck.true(label+': coupling held exactly fixed', solution.alpha == reference.alpha)
        ck.close(label+': solved outer scalar remains 0.2', solution.phip, .2, 1e-8)
        ck.close(label+': same normalized product', solution.product, reference.product, 1e-9)
        exterior = exterior_check(solution)
        ck.true(label+': positive outer slope and asymptotically flat exterior diagnostics',
                solution.bdp > 0 and exterior['positive_asymptotic_slope'])
        ck.small(label+': exterior Hamiltonian', exterior['hamiltonian_max_abs'], 1e-8)
        ck.small(label+': exterior slope stabilizes between r=1e4 and 1e5',
                 np.ptp(exterior['lambda_values']), 1e-6)
        ck.true(label+': r phi-prime decays in the far exterior',
                abs(exterior['scaled_phi_derivative_values'][-1]) < .2*abs(exterior['scaled_phi_derivative_values'][0]))
        ck.true(label+': strict field-equation entropy-gap bound respected',
                d['entropy_gap_actual_over_Sstar'] > d['entropy_gap_lower_over_Sstar'] > 0)
        d['exterior'] = exterior
        fixed_rows.append(d)
        write_json(directory/d['profile_file'], profile_record(solution))
    exact_rows = [exact_benchmark(mass, phi_inf, directory) for mass, phi_inf in ((1.6, .2), (3., -.45))]
    output = dict(schema_version=1, equations='bsecond=-b phiprime^2; (P phiprime)prime=alpha sinh(alpha phi)/b^2; W=2cosh(alpha phi)',
                  horizons=dict(r_plus=RP, r_minus=RM), charges=[1., 1.],
                  seed_provenance='analytic RN and Legendre data only; no inward-shooting import or numerical seed',
                  convergence=rows, strict_setting_spreads=dict(alpha=alpha_spread, normalized_product=product_spread),
                  fixed_alpha=reference.alpha, fixed_theory_rows=fixed_rows,
                  fixed_theory_seed_preparation=preparation_rows, exact_controls=exact_rows,
                  limitations=['Empirical convergence errors, not rigorous interval bounds.',
                               'Regularity enforced through order-varied local Taylor expansions; not an analytic existence theorem.',
                               'Finite-radius exterior diagnostics support asymptotic flatness; they are not a global proof.',
                               'The fixed-theory sampled solutions do not by themselves prove global connectedness or maximal continuation.'],
                  environment=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__),
                  elapsed_seconds=time.perf_counter()-started)
    write_json(directory/'tuned_branch_bvp.json', output)
    print('\nIndependent two-horizon collocation: alpha=%.15f, product=%.15f, deficit=%.12g' %
          (reference.alpha, reference.product, 1-reference.product))
    print('Raw convergence, constraints, derivatives, curvature and profiles:', directory)
    ck.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path,
                        default=Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1]/"results/data"))/"bvp")
    args = parser.parse_args()
    run(args.output_dir)
