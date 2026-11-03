r"""Independent exact-solution benchmark for the nonlinear radial integrator.

The oracle is the Case 1 solution of GJN2014, arXiv:1410.3478,
Appendix A.1, (A.1)--(A.4), with Q1 = Q2 = 1 and alpha = (2, -2):
  Sigma = -sinh(2 phi_inf)/M,
  Delta^2 = M^2 + Sigma^2 - 2 cosh(2 phi_inf),
  b^2 = R^2 - Sigma^2,  phi = phi_inf + atanh(Sigma/R).
Its asymptotically normalized coordinates (T,R) map to the integrator by
  L = 2 Delta/(RP-RM),  R = M + L [r-(RP+RM)/2],  T = t/L.
Consequently a^2 = [(R-M)^2-Delta^2]/(L^2 b^2) = P(r)/b^2;
the charges are unchanged, b'_infinity = L, and d/dr = L d/dR.

This file does not import dilaton_exact.py or use numerical solutions as its
reference. It tests all four evolved fields on both sides of the outer horizon,
the horizon/launch data, the finite-radius exterior API, and the inferred inner
horizon data for three nonconstant solutions with different masses and moduli.
Near r-, a roundoff-sized spurious logarithmic coefficient is amplified in phi'
by 1/(r-r-). We therefore test phi' directly away from the endpoint and test
(r-r-) phi' throughout the interior, alongside b, b', and phi themselves.
"""
import numpy as np
from scipy.integrate import solve_ivp

import checklib as ck
import inner_regularity_physical as ip


class ExactDilaton:
    def __init__(self, mass, phi_inf):
        self.mass, self.phi_inf = mass, phi_inf
        self.sigma = -np.sinh(2 * phi_inf) / mass
        self.delta = np.sqrt(mass ** 2 + self.sigma ** 2 - 2 * np.cosh(2 * phi_inf))
        self.scale = 2 * self.delta / (ip.RP - ip.RM)

    def radius(self, r):
        return self.mass + self.scale * (np.asarray(r) - (ip.RP + ip.RM) / 2)

    def fields(self, r):
        R = self.radius(r)
        b2 = R ** 2 - self.sigma ** 2
        b = np.sqrt(b2)
        return np.array([b, self.scale * R / b,
                         self.phi_inf + np.arctanh(self.sigma / R),
                         -self.scale * self.sigma / b2])

    def horizon_derivatives(self):
        b, bd, _, phd = self.fields(ip.RP)
        phdd = 2 * self.scale ** 2 * self.sigma * self.radius(ip.RP) / b ** 4
        bdd = -self.scale ** 2 * self.sigma ** 2 / b ** 3
        return np.array([bd, phd, phdd, bdd])


def scaled_error(actual, expected):
    """Absolute error for small fields, relative error for fields larger than one."""
    return (np.asarray(actual) - expected) / np.maximum(1.0, np.abs(expected))


def benchmark(mass, phi_inf):
    exact = ExactDilaton(mass, phi_inf)
    tag = 'M=%g, phi_inf=%g' % (mass, phi_inf)
    bp, _, php, _ = exact.fields(ip.RP)
    args = (2.0, -2.0, bp, php)
    ck.true(tag + ': exact inner horizon lies outside the curvature singularity',
            exact.mass - exact.delta > abs(exact.sigma))
    ck.small_all(tag + ': outer-horizon derivatives agree with the closed form',
                 scaled_error(ip.horizon_data(*args), exact.horizon_derivatives()), 2e-12)
    for side in (-1, 1):
        ck.small_all(tag + ': Taylor launch agrees with the exact profile (side %+d)' % side,
                     scaled_error(ip.launch(*args, 1e-6, side), exact.fields(ip.RP + side * 1e-6)), 2e-11)

    interior = ip.inward(*args)
    offsets = np.geomspace(ip.REND, ip.RP - ip.RM - 1e-6, 150)
    radii = ip.RM + offsets
    values = ip.sample_inward(interior, radii, 'exact-dilaton interior benchmark')
    reference = exact.fields(radii)
    for k, name in enumerate(('b', "b'", 'phi')):
        ck.small_all(tag + ': interior %s profile, down to r-r-=1e-8' % name,
                     scaled_error(values[k], reference[k]), 1e-9)
    ck.small_all(tag + ": interior phi' profile for r-r- >= 1e-4",
                 scaled_error(values[3, offsets >= 1e-4], reference[3, offsets >= 1e-4]), 1e-8)
    ck.small_all(tag + ": interior (r-r-) phi' profile, down to r-r-=1e-8",
                 offsets * (values[3] - reference[3]), 1e-10)

    # Exercise the production RHS and launch independently of its outward()
    # summary, then compare that public summary to the same exact endpoint.
    start, end = ip.RP + 1e-6, 1e5
    exterior = solve_ivp(ip.rhs_factory(2.0, -2.0), [start, end], ip.launch(*args, 1e-6, 1),
                         method='DOP853', rtol=1e-12, atol=1e-14, dense_output=True)
    radii = ip.RP + np.geomspace(1e-6, end - ip.RP, 150)
    valid = (exterior.success and exterior.status == 0 and np.all(np.isfinite(exterior.t))
             and np.all(np.diff(exterior.t) > 0) and np.all(np.isfinite(exterior.y))
             and exterior.t[0] <= radii[0] and exterior.t[-1] >= radii[-1])
    ck.true(tag + ': exterior integration completes and covers its profile grid', valid)
    if not valid:
        return
    values, reference = exterior.sol(radii), exact.fields(radii)
    for k, name in enumerate(('b', "b'", 'phi', "phi'")):
        ck.small_all(tag + ': exterior %s profile out to r=1e5' % name,
                     scaled_error(values[k], reference[k]), 1e-9)
    outside = ip.outward(*args)
    ck.true(tag + ': production exterior reports the exact asymptotically flat branch', outside['flat'])
    ck.small_all(tag + ': production exterior endpoint matches the finite-radius exact fields',
                 scaled_error(np.array([outside['lambda_inf'], outside['phi_inf']]),
                              exact.fields(end)[[1, 2]]), 1e-9)

    coefficient, bm = ip.two_radius(interior, 1e-7, 1e-8)
    ck.small(tag + ': inferred logarithmic coefficient vanishes', coefficient, 1e-10)
    ck.close(tag + ': inferred inner radius matches its exact horizon value', bm, exact.fields(ip.RM)[0], 1e-10)
    ck.close(tag + ': exact normalized horizon area product equals one',
             bp ** 2 * exact.fields(ip.RM)[0] ** 2 / 4, 1.0, 2e-12)
    diagnosis = ip.diagnose(*args)
    ck.true(tag + ': production diagnostic identifies the known regular solution at resolution',
            diagnosis['outcome'] == 'zero', ip.fmt_c(diagnosis))
    ck.close(tag + ': diagnosed area product matches the exact result',
             diagnosis['product'] if diagnosis['product'] is not None else np.nan, 1.0, 1e-9)

    # A missing radial normalization or a reversed scalar-charge sign must be
    # visible to the same independent profile comparison used above.
    if phi_inf > 0.3:
        ck.large_any(tag + ': profile oracle detects a missing radial derivative scale',
                     values[[1, 3]] - reference[[1, 3]] / exact.scale, 1e-2)
        wrong_scalar = 2 * exact.phi_inf - reference[2]
        ck.large_any(tag + ': profile oracle detects a reversed scalar-charge sign',
                     values[2] - wrong_scalar, 1e-2)


if __name__ == '__main__':
    for mass, phi_inf in ((1.6, 0.2), (2.0, 0.35), (3.0, -0.45)):
        benchmark(mass, phi_inf)
    ck.finish()
