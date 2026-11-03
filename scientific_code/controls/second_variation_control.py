#!/usr/bin/env python3
"""Fixed-charge entropy Hessian in scalar/gauge directions of the static throat.

This evaluates the entropy functional, not a Lorentzian perturbation energy.
The metric is fixed, the static Maxwell fields are eliminated at fixed fluxes,
and k=J=0 remains true in the separated magnetic/electric two-field sector.
The BVP profile is computed afresh; endpoint-order/mesh comparisons are numerical
error diagnostics and do not replace the paper's local analytic existence proof.

Run from the archive: python controls/second_variation_control.py
Elsewhere use --source-root PATH (directory containing scripts/bvp_collocation.py).
Writes controls/second_variation.json below HOT_ATTRACTORS_OUTPUT, which defaults
to results/data in the package.
"""
from pathlib import Path
import argparse
import json
import os
import sys
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import Legendre, leggauss


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def spherical_controls():
    z = sp.Symbol('z', real=True)
    answers = {}
    for degree in (0, 1, 2, 3, 4):
        chi = sp.legendre(degree, z)
        gradient = sp.integrate((1-z*z)*sp.diff(chi, z)**2, (z, -1, 1))
        moment = sp.integrate(chi, (z, -1, 1))
        square = sp.integrate(chi*chi, (z, -1, 1))
        curvature = sp.simplify(gradient+12*(moment*moment-square))
        expected = 24 if degree == 0 else sp.Rational(2, 2*degree+1)*(degree*(degree+1)-12)
        require(curvature == expected, 'spherical Legendre Hessian formula')
        answers[str(degree)] = str(curvature)
    # Distinct sign controls and known zero-mode limits are mandatory.
    require(sp.Rational(answers['1']) < 0 < sp.Rational(answers['4']), 'two opposite Hessian signs')
    require(answers['3'] == '0', 'nu=3 spherical bifurcation mode')
    flat = sp.integrate((1-z*z)-2*z*z, (z, -1, 1))
    require(flat == 0, 'alpha=2 P1 flat tangent at the spherical member')
    # Independent differentiation of a quotient validates the nonlocal terms.
    sigma, I, J, K, alpha = sp.symbols('sigma I J K alpha', real=True)
    for sign in (1, -1):
        inverse = 1/(I+sign*alpha*J*sigma+alpha**2*K*sigma**2/2)
        actual = sp.diff(inverse, sigma, 2).subs(sigma, 0)
        require(sp.simplify(actual-alpha**2*(2*J**2/I**3-K/I**2)) == 0,
                'second derivative of inverse electric/magnetic integral')
    return answers


def balanced_fields(solution, z, RM, RP):
    delta = RP-RM
    radius = (RP+RM)/2+delta*z/2
    x = (radius-RM)/delta
    ratio = solution.bm/solution.bp
    weight = 1+(ratio-1)*x
    mapped = RM+delta*ratio*x/weight
    b, bd, phi, pd = solution.sample(mapped)
    return weight*b/np.sqrt(ratio), phi, pd*ratio/weight**2*delta/2


def evaluate_profile(solution, RM, RP, count):
    z, quadrature = leggauss(count)
    b, phi, phiz = balanced_fields(solution, z, RM, RP)
    area_scale = float(solution.bp*solution.bm)
    alpha = float(solution.alpha)
    # Exact continued-charge formulas for Q=P=1.
    p, q = area_scale/2, -area_scale/2
    base = area_scale/b**2
    weights = [base*np.exp(sign*alpha*phi) for sign in (1, -1)]
    integrals = [float(quadrature@weight) for weight in weights]
    require(max(abs(value-area_scale) for value in integrals) < 2e-8,
            'Maxwell integrals agree with regular two-horizon identities')
    result = {}
    for degree in (1, 3, 4):
        poly = Legendre.basis(degree)
        chi, derivative = poly(z), poly.deriv()(z)
        gradient = float(quadrature@((1-z*z)*derivative**2))
        moments = [float(quadrature@(weight*chi)) for weight in weights]
        squares = [float(quadrature@(weight*chi**2)) for weight in weights]
        gauge = 0.
        for charge, integral, moment, square in zip((p,q), integrals, moments, squares):
            gauge += (2*alpha**2/area_scale)*charge**2*(2*moment**2/integral**3-square/integral**2)
        hessian = gradient+gauge
        first_variation = float(quadrature@((1-z*z)*phiz*derivative))+(2*alpha/area_scale)*(-p*p*moments[0]/integrals[0]**2+q*q*moments[1]/integrals[1]**2)
        require(abs(first_variation) < 1e-8, 'background is stationary along the fixed-charge scalar/Maxwell path')
        # Direct finite differences of the reduced entropy / (pi*area_scale),
        # including the scalar kinetic term and varying e,u to conserve flux.
        def reduced_entropy(displacement):
            displaced = phi+displacement*chi
            displaced_z = phiz+displacement*derivative
            ia = float(quadrature@(base*np.exp(alpha*displaced)))
            ib = float(quadrature@(base*np.exp(-alpha*displaced)))
            # Constructed profiles have fixed magnetic/electric flux exactly.
            magnetic_density = 2*p*base*np.exp(alpha*displaced)/ia
            e = 2*q/ib
            magnetic_charge = float(quadrature@magnetic_density)/2
            electric_charge = e*float(quadrature@(base*np.exp(-alpha*displaced)))/2
            require(abs(magnetic_charge-p) < 1e-12 and abs(electric_charge-q) < 1e-12,
                    'off-shell scalar path preserves both continued charges')
            return float(.5*(quadrature@((1-z*z)*displaced_z**2))
                         +(2/area_scale)*(p*p/ia+q*q/ib))
        central = reduced_entropy(0.)
        finite_differences = []
        for step in (.001, .0005):
            finite_differences.append((reduced_entropy(step)+reduced_entropy(-step)-2*central)/step**2)
        require(abs(finite_differences[-1]-hessian) < 4e-5,
                'analytic Hessian agrees with direct entropy finite differences')
        result[str(degree)] = dict(hessian_over_pi_B=hessian, first_variation_over_pi_B=first_variation, gradient=gradient,
                                  electromagnetic_part=gauge,
                                  finite_difference_steps=[.001,.0005],
                                  finite_difference_values=finite_differences)
    require(result['1']['hessian_over_pi_B'] < -5., 'P1 remains a strictly negative direction')
    require(result['4']['hessian_over_pi_B'] > 1., 'P4 remains a strictly positive direction')
    return dict(quadrature_nodes=count,alpha=alpha,B=area_scale,
                p=p,q=q,maxwell_integrals=integrals,modes=result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    source = args.source_root.resolve()
    require((source/'scripts/bvp_collocation.py').is_file(), 'source root must contain scripts/bvp_collocation.py')
    sys.path.insert(0, str(source/'scripts'))
    from bvp_collocation import solve_branch, RM, RP
    symbolic = spherical_controls()
    records = []
    for order, initial_nodes in ((6,401),(8,601)):
        solution = solve_branch(order=order,cutoff=1e-4,tolerance=1e-10,initial_nodes=initial_nodes)
        diagnostics = solution.diagnostics()
        require(diagnostics['off_grid_scaled_ode_max'] < 1e-8, 'fresh BVP off-grid residual')
        coarse = evaluate_profile(solution, RM, RP, 160)
        fine = evaluate_profile(solution, RM, RP, 320)
        for degree in ('1','3','4'):
            require(abs(coarse['modes'][degree]['hessian_over_pi_B']-fine['modes'][degree]['hessian_over_pi_B']) < 1e-8,
                    'quadrature-node refinement stability')
        fine.update(endpoint_order=order,cutoff=1e-4,initial_nodes=initial_nodes,
                    off_grid_scaled_ode_max=diagnostics['off_grid_scaled_ode_max'])
        records.append(fine)
    for degree in ('1','3','4'):
        require(abs(records[0]['modes'][degree]['hessian_over_pi_B']-records[1]['modes'][degree]['hessian_over_pi_B']) < 1e-8,
                'independent endpoint-order stability')
    report = dict(status='PASS',spherical_hessians_over_pi_B=symbolic,finite_amplitude=records,
                  conclusion='The fixed-charge static entropy functional has both signs in admissible scalar/Maxwell directions.',
                  scope='No Lorentzian dynamical stability conclusion. Finite-amplitude signs are numerical; small-amplitude saddle follows analytically by continuity from P1 and P4.',
                  environment=dict(python=sys.version.split()[0],numpy=np.__version__,sympy=sp.__version__))
    encoded = json.dumps(report,indent=2)+'\n'
    directory = Path(os.environ.get('HOT_ATTRACTORS_OUTPUT', source/'results/data'))/'controls'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/'second_variation.json').write_text(encoded)
    print(encoded,end='')


if __name__ == '__main__':
    main()
