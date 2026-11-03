r"""Independent double-shooting validation of the two-horizon collocation branch.

This uses separately coded cubic horizon expansions, joins two regular initial
value integrations at the midpoint, and imports no inward-shooting code.  Its
initial values are sqrt(24), b-=2/b+, phi-=-phi+: the linear RN/Legendre limit,
not the tuned numerical root.  We match b,phi,phi' and leave b' as an independent
first-integral check.  Thus it differs from simultaneous global collocation in
its algorithm, Taylor order, variables, boundary implementation and seed.

The reported convergence is numerical evidence, not a rigorous existence proof.
"""
from copy import copy
from pathlib import Path
import os
from types import SimpleNamespace
from unittest.mock import patch
import json
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import root

import checklib as ck
import bvp_collocation as bc

RP, RM = 1.0, 0.35
D = RP - RM
BP = 2.5*(D+np.sqrt(D*D+8))/2
PHIP = 0.2
MID = (RP+RM)/2


def cubic_fields(h, b, phi, alpha, offset):
    """Independent derivative formulas through third order, signed offset."""
    d = 2*h-RP-RM
    w0 = 2*np.cosh(alpha*phi)
    w1, w2 = 2*alpha*np.sinh(alpha*phi), alpha**2*w0
    w3 = alpha**2*w1
    v, u = (b-w0/b)/d, w1/(2*b*b*d)
    b2 = -b*u*u
    s1 = w2*u/(2*b*b)-w1*v/b**3
    f2 = (s1-2*u)/(2*d)
    b3 = -v*u*u-2*b*u*f2
    s2 = (w3*u*u+w2*f2)/(2*b*b)-2*w2*u*v/b**3-w1*b2/b**3+3*w1*v*v/b**4
    f3 = (s2-6*f2)/(3*d)
    e = np.asarray(offset)
    return np.asarray([b+v*e+b2*e*e/2+b3*e**3/6, v+b2*e+b3*e*e/2,
                       phi+u*e+f2*e*e/2+f3*e**3/6, u+f2*e+f3*e*e/2])


def rhs(alpha):
    def evaluate(r, y):
        b, v, phi, u = y
        p, dp = (r-RP)*(r-RM), 2*r-RP-RM
        return [v, -b*u*u, u, (alpha*np.sinh(alpha*phi)/(b*b)-dp*u)/p]
    return evaluate


def validate_half(solution, start, end):
    """Do not evaluate dense output from a failed, partial or nonfinite solve."""
    if not solution.success or solution.status != 0:
        raise RuntimeError('independent half integration failed')
    t, y = np.asarray(solution.t), np.asarray(solution.y)
    direction = np.sign(end-start)
    if (t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t))
            or not np.all(direction*np.diff(t) > 0) or t[0] != start or t[-1] != end
            or y.shape != (4, t.size) or not np.all(np.isfinite(y))
            or not callable(getattr(solution, 'sol', None))):
        raise RuntimeError('independent half integration is incomplete or nonfinite')


def halves(parameters, cutoff, rtol):
    alpha, bm, phim = parameters
    left_start, right_start = RM+cutoff, RP-cutoff
    left = solve_ivp(rhs(alpha), (left_start, MID), cubic_fields(RM, bm, phim, alpha, cutoff),
                     method='DOP853', rtol=rtol, atol=rtol/100, dense_output=True)
    right = solve_ivp(rhs(alpha), (right_start, MID), cubic_fields(RP, BP, PHIP, alpha, -cutoff),
                      method='DOP853', rtol=rtol, atol=rtol/100, dense_output=True)
    validate_half(left, left_start, MID)
    validate_half(right, right_start, MID)
    return left, right


def double_shoot(cutoff=1e-7, rtol=2e-13):
    def mismatch(parameters):
        left, right = halves(parameters, cutoff, rtol)
        values = (left.y[:, -1]-right.y[:, -1])[[0, 2, 3]]
        ck.max_abs(values)
        return values
    fit = root(mismatch, [np.sqrt(24), 2/BP, -PHIP], tol=1e-12)
    residual = mismatch(fit.x)
    if not fit.success or not np.all(np.isfinite(fit.x)) or ck.max_abs(residual) > 1e-10:
        raise RuntimeError('independent midpoint match did not converge')
    alpha, bm, phim = fit.x
    left, right = halves(fit.x, cutoff, rtol)
    join = left.y[:, -1]-right.y[:, -1]

    def sample(radii):
        scalar = np.ndim(radii) == 0
        r = np.atleast_1d(np.asarray(radii, dtype=float))
        if not np.all(np.isfinite(r)) or np.any(r < RM) or np.any(r > RP):
            raise RuntimeError('independent profile requested outside horizons')
        values = np.empty((4, r.size))
        inner, outer = r < RM+cutoff, r > RP-cutoff
        middle_left = (~inner) & (r <= MID)
        middle_right = (~outer) & (r > MID)
        if np.any(inner):
            values[:, inner] = cubic_fields(RM, bm, phim, alpha, r[inner]-RM)
        if np.any(outer):
            values[:, outer] = cubic_fields(RP, BP, PHIP, alpha, r[outer]-RP)
        if np.any(middle_left):
            values[:, middle_left] = left.sol(r[middle_left])
        if np.any(middle_right):
            values[:, middle_right] = right.sol(r[middle_right])
        ck.max_abs(values)
        if np.any(values[0] <= 0):
            raise RuntimeError('independent profile has nonpositive area radius')
        return values[:, 0] if scalar else values

    grid = np.r_[np.linspace(RM+cutoff, MID, 401), np.linspace(MID, RP-cutoff, 401)]
    b, v, phi, u = sample(grid)
    p, dp = (grid-RP)*(grid-RM), 2*grid-RP-RM
    constant = -b*b+dp*b*v-p*v*v+2*np.cosh(alpha*phi)-p*b*b*u*u
    horizon = sample(RM)
    w = 2*np.cosh(alpha*phim)/bm**2
    record = dict(cutoff=float(cutoff), rtol=float(rtol), alpha=float(alpha), bminus=float(bm),
                  phiminus=float(phim), product=float(BP**2*bm**2/4),
                  midpoint_mismatch=join.tolist(), max_b2H=float(ck.max_abs(constant)),
                  inner_b_prime=float(horizon[1]), inner_phi_prime=float(horizon[3]),
                  inner_kretschmann=float((12-24*w+20*w*w)/bm**4),
                  seed='analytic sqrt(24), bminus=2/bplus, phiminus=-phiplus')
    return record, sample, (left, right)


def curvature(r, fields):
    """Geometric spherical invariant, with b'' supplied by its field equation."""
    b, v, phi, u = fields
    p, dp = (np.asarray(r)-RP)*(np.asarray(r)-RM), 2*np.asarray(r)-RP-RM
    bdd = -b*u*u
    a = p/b**2
    ad = dp/b**2-2*p*v/b**3
    add = 2/b**2-4*dp*v/b**3-2*p*bdd/b**3+6*p*v*v/b**4
    value = add**2+2*(ad*v/b)**2+2*((2*a*bdd+ad*v)/b)**2+4*(1-a*v*v)**2/b**4
    ck.max_abs(value)
    return value


def finite_difference_curvature(sample, radii, step=2e-5):
    """Differentiate the metric directly; do not substitute matter equations."""
    r = np.asarray(radii)
    values = [sample(r+k*step) for k in (-2, -1, 0, 1, 2)]
    b = np.asarray([v[0] for v in values])
    aa = np.asarray([((r+k*step-RP)*(r+k*step-RM))/b[i]**2
                     for i, k in enumerate((-2, -1, 0, 1, 2))])
    bd = (b[0]-8*b[1]+8*b[3]-b[4])/(12*step)
    bdd = (-b[4]+16*b[3]-30*b[2]+16*b[1]-b[0])/(12*step**2)
    ad = (aa[0]-8*aa[1]+8*aa[3]-aa[4])/(12*step)
    add = (-aa[4]+16*aa[3]-30*aa[2]+16*aa[1]-aa[0])/(12*step**2)
    a, center = aa[2], b[2]
    result = add**2+2*(ad*bd/center)**2+2*((2*a*bdd+ad*bd)/center)**2+4*(1-a*bd*bd)**2/center**4
    ck.max_abs(result)
    return result


def rejects(label, callback, exception=(RuntimeError, bc.BVPError, ValueError)):
    try:
        callback()
    except exception:
        ck.true(label, True)
    else:
        ck.true(label, False)


def run_checks():
    solution = bc.solve_branch()
    rows, samplers = [], []
    for cutoff in (1e-5, 1e-6, 1e-7):
        row, sample, parts = double_shoot(cutoff)
        rows.append(row)
        samplers.append(sample)
        ck.small_all('independent double shoot %.0e: three midpoint matches' % cutoff,
                     np.asarray(row['midpoint_mismatch'])[[0, 2, 3]], 2e-11)
        ck.true('independent double shoot %.0e: finite positive radius' % cutoff,
                row['bminus'] > 0 and row['inner_kretschmann'] > 0)
    finest, fine_sample = rows[-1], samplers[-1]
    ck.small('unmatched midpoint b prime agrees independently', finest['midpoint_mismatch'][1], 1e-10)
    ck.small('independent Hamiltonian constant is zero throughout interior', finest['max_b2H'], 1e-10)
    for key in ('alpha', 'bminus', 'product', 'inner_b_prime', 'inner_phi_prime', 'inner_kretschmann'):
        differences = [abs(rows[i+1][key]-rows[i][key]) for i in (0, 1)]
        ck.true('cubic endpoint refinement converges: '+key, differences[1] < differences[0]/100)
        ck.small('finest independent endpoint change: '+key,
                 differences[-1]/max(1, abs(finest[key])), 5e-9)
    ck.close('collocation coupling agrees with independently joined horizons', solution.alpha, finest['alpha'], 2e-9)
    ck.close('collocation product agrees with independently joined horizons', solution.product, finest['product'], 2e-10)
    ck.small_all('both horizon field derivatives agree between methods',
                 (solution.sample([RM, RP])-fine_sample([RM, RP]))/np.maximum(1, np.abs(fine_sample([RM, RP]))), 2e-8)
    grid = RM+D*(np.arange(181)+np.sqrt(2)/3)/182
    first, second = solution.sample(grid), fine_sample(grid)
    for i, label in enumerate(('b', 'b prime', 'phi', 'phi prime')):
        ck.small_all('held-out interior profile agrees: '+label,
                     (first[i]-second[i])/np.maximum(1, np.abs(second[i])), 2e-8)
    kcoll, kind = curvature(grid, first), curvature(grid, second)
    ck.small_all('held-out interior Kretschmann profiles agree', (kcoll-kind)/np.maximum(1, np.abs(kind)), 2e-8)
    fd_grid = np.linspace(RM+0.0031, RP-0.0047, 41)
    kfd = finite_difference_curvature(solution.sample, fd_grid)
    expected = curvature(fd_grid, solution.sample(fd_grid))
    ck.small_all('direct metric differentiation verifies curvature independently of matter substitution',
                 (kfd-expected)/np.maximum(1, np.abs(expected)), 3e-5)
    ck.true('resolved product deficit exceeds cutoff variation by at least a million',
            1-finest['product'] > 1e6*abs(rows[-2]['product']-finest['product']))
    # The untuned linear RN seed solves neither nonlinear matching nor inner regularity.
    left, right = halves([np.sqrt(24), 2/BP, -PHIP], 1e-7, 2e-13)
    ck.large_any('negative control: untuned linear seed has resolved midpoint mismatch',
                 left.y[:, -1]-right.y[:, -1], 1e-4)
    # Exact nonconstant alpha=2 dilaton fields validate the independently coded
    # cubic Taylor formula at BOTH horizons and on BOTH signed sides.
    mass, phi_inf = 2.0, 0.35
    sigma = -np.sinh(2*phi_inf)/mass
    delta = np.sqrt(mass**2+sigma**2-2*np.cosh(2*phi_inf))
    scale = 2*delta/D
    def exact(r):
        R = mass+scale*(np.asarray(r)-(RP+RM)/2)
        b = np.sqrt(R*R-sigma*sigma)
        return np.asarray([b, scale*R/b, phi_inf+np.arctanh(sigma/R), -scale*sigma/(b*b)])
    for h in (RM, RP):
        bh, _, ph, _ = exact(h)
        for sign in (-1, 1):
            e = sign*1e-6
            ck.small_all('independent cubic launch matches exact dilaton at h=%g, side=%+d' % (h, sign),
                         (cubic_fields(h, bh, ph, 2, e)-exact(h+e))/np.maximum(1, np.abs(exact(h+e))), 2e-10)
    # Check the geometry formula on RN without invoking any horizon identity.
    R = np.linspace(2.1, 5, 21)
    m, q2 = (RP+RM)/2, RP*RM
    fields = np.asarray([R, np.ones_like(R), np.zeros_like(R), np.zeros_like(R)])
    target = 48*m*m/R**6-96*m*q2/R**7+56*q2*q2/R**8
    ck.small_all('geometric Kretschmann formula reproduces full RN radial invariant', curvature(R, fields)-target, 2e-14)

    # Fail-closed tests cover invalid and incomplete solver output.
    bad = copy(parts[0]); bad.success = False; bad.status = -1
    rejects('independent sampler refuses failed IVP', lambda: validate_half(bad, RM+1e-7, MID))
    bad = copy(parts[0]); bad.t = bad.t[:-1]; bad.y = bad.y[:, :-1]
    rejects('independent sampler refuses success-marked partial IVP', lambda: validate_half(bad, RM+1e-7, MID))
    bad = copy(parts[0]); bad.y = bad.y.copy(); bad.y[0, 0] = np.nan
    rejects('independent sampler refuses NaN trajectory before reduction', lambda: validate_half(bad, RM+1e-7, MID))
    rejects('collocation rejects an actual exhausted node budget',
            lambda: bc.solve_branch(initial_nodes=20, max_nodes=20))
    bad = copy(solution.collocation); bad.y = bad.y.copy(); bad.y[0, 0] = np.nan
    with patch.object(bc, 'solve_bvp', return_value=bad):
        rejects('collocation rejects success-marked NaN field data', lambda: bc.solve_branch())
    bad = copy(solution.collocation); bad.rms_residuals = bad.rms_residuals.copy(); bad.rms_residuals[0] = np.nan
    with patch.object(bc, 'solve_bvp', return_value=bad):
        rejects('collocation rejects NaN residual before maximum comparison', lambda: bc.solve_branch())
    bad = copy(solution.collocation); bad.x = bad.x[1:]; bad.y = bad.y[:, 1:]
    with patch.object(bc, 'solve_bvp', return_value=bad):
        rejects('collocation rejects success-marked partial interval', lambda: bc.solve_branch())
    with patch.object(bc, 'solve_bvp', return_value=solution.collocation), \
            patch.object(bc, 'taylor_fields', return_value=np.full(4, np.nan)):
        rejects('collocation rejects NaN boundary residual before maximum comparison', lambda: bc.solve_branch())
    truncated = copy(solution)
    truncated.collocation = copy(solution.collocation)
    truncated.collocation.x = solution.collocation.x[10:]
    rejects('dense sampling refuses to clip a genuinely truncated domain',
            lambda: truncated.sample(RM+1.1*solution.cutoff))
    for invalid in ([RM-1e-6], [RP+1e-6], [np.nan]):
        rejects('collocation sampling rejects invalid radius '+str(invalid), lambda r=invalid: solution.sample(r))
    # The exterior solver must reject successful but incomplete output BEFORE
    # querying dense output at the large radii used in its AF diagnostics.
    partial = SimpleNamespace(success=True, status=0, t=np.array([RP+1e-5, 2.0]),
                              y=np.ones((4, 2)), sol=lambda r: np.ones((4, np.size(r))))
    with patch.object(bc, 'solve_ivp', return_value=partial):
        rejects('exterior rejects success-marked short integration', lambda: bc.exterior_check(solution))
    invalid_exterior = SimpleNamespace(success=True, status=0, t=np.array([RP+1e-5, 1e5]),
                                      y=np.ones((4, 2)), sol=lambda r: np.full((4, np.size(r)), np.nan))
    with patch.object(bc, 'solve_ivp', return_value=invalid_exterior):
        rejects('exterior rejects NaN dense data before AF comparisons', lambda: bc.exterior_check(solution))
    invalid_exterior = SimpleNamespace(success=True, status=0, t=np.array([RP+1e-5, np.nan, 1e5]),
                                      y=np.ones((4, 3)), sol=lambda r: np.ones((4, np.size(r))))
    with patch.object(bc, 'solve_ivp', return_value=invalid_exterior):
        rejects('exterior rejects NaN integration mesh before dense evaluation', lambda: bc.exterior_check(solution))
    rejects('exterior refuses a radius that would place diagnostic samples outside its domain',
            lambda: bc.exterior_check(solution, rmax=5))
    record = dict(method='independent two-sided cubic Taylor shooting joined at midpoint',
                  radial_horizons=[RM, RP], bplus=BP, phiplus=PHIP, refinements=rows,
                  collocation_alpha=solution.alpha, collocation_product=solution.product,
                  held_out_radii=grid.tolist(), collocation_kretschmann=kcoll.tolist(),
                  independent_kretschmann=kind.tolist(),
                  direct_metric_curvature_scaled_error=float(ck.max_abs((kfd-expected)/np.maximum(1, np.abs(expected)))),
                  interpretation='empirical convergence and independent algorithm agreement, not rigorous error bounds')
    destination = Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1]/"results/data"))/"bvp"/"independent_double_shoot.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(record, indent=2)+'\n')
    print('Independent alpha=%.14f, product=%.14f, Kminus=%.9f' %
          (finest['alpha'], finest['product'], finest['inner_kretschmann']))


if __name__ == '__main__':
    run_checks()
    ck.finish()
