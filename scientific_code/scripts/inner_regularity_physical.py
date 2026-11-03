r"""
inner_regularity_physical.py -- inner-horizon regularity of static hairy black holes, with PHYSICAL data.

Model: V_eff = Q1^2 e^{alpha1 phi} + Q2^2 e^{alpha2 phi} (Q1 = Q2 = 1), kinetic term 2 (dphi)^2, V = 0, D = 4.
Exactly a^2 b^2 = P(r) = (r - r+)(r - r-); remaining equations
    b'' = -b phi'^2,      (P phi')' = V_eff'(phi) / (2 b^2),
    horizon constraint at r+:   (r+ - r-) b'(r+)/b(r+) = 1 - V_eff(phi+)/b(r+)^2 .
Outer-horizon data (b+, phi+) are taken PHYSICAL: b+^2 > V_eff(phi+) so that b'(r+) > 0, and we check by integrating
OUTWARDS that the exterior is asymptotically flat (b' -> lambda > 0, phi -> phi_inf).  We then integrate INWARDS
through region II, down to r - r- = 1e-8 (double precision resolves r - r- only to ~1e-16 relative to r-), and
diagnose the inner horizon through
    c = lim_{r->r-} (r - r-) phi'(r)      (c = 0 iff there is no log(r - r-) term).
diagnose() gives one of three outcomes -- 'nonzero' (a resolved logarithmic term: the inner horizon is singular),
'zero' (consistent with a regular inner horizon at the stated resolution) or 'unresolved' -- from nested two-radius
fits, their observed rate of convergence and the sensitivity to the launch distance from r+.  Horizon quantities at
r- (the product b+^2 b-^2/V_*^2) are evaluated only for the outcome 'zero'.
The reference point is the doubly-extremal solution: phi = phi_*, b = r + beta with (r+ + beta)(r- + beta) = V_*.

Also checked here:
  * second order: c = c2 eps^2 + O(eps^3) with c2 = (-1)^nu V3(phi_*)/(8 V_*) int_{-1}^{1} P_nu^3 dz,
    V3 = third derivative of V_eff (vanishes for odd nu by the z -> -z parity, and for A2 because V3(phi_*) = 0);
  * third order on the nu = 3 family alpha1 alpha2 = -24: c3 = 24 (s^2 - 4)/1001, s = alpha1 + alpha2 (c3_exact.py);
  * tuned branches (Davydov, arXiv:1711.04198): regular inner horizons at NON-integer nu and discrete horizon scalar,
    bifurcating from nu = 3; S+S- is constant along such a branch but differs from S_ext^2;
  * small signals: tiny nonzero c are classified as 'nonzero', not as regular;
  * singular case: near r-, b ~ (r - r-)^p with p(1-p) = c^2, i.e. a Kasner singularity.
Run:  python3 inner_regularity_physical.py   (included in the full mode; runtime is hardware dependent)
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import minimize_scalar, brentq
from scipy.special import eval_legendre
import checklib as ck

RP, RM = 1.0, 0.35
REND = 1e-8                  # inward integrations stop at r - r- = REND
FITS = (1e-4, 1e-5, 1e-6, 1e-7, 1e-8)   # nested two-radius fits (1e-4,1e-5), (1e-5,1e-6), (1e-6,1e-7), (1e-7,1e-8)


class IntegrationError(RuntimeError):
    """An inward solution cannot support the requested numerical evaluation."""


def validate_inward(sol, radii=(), context='inward integration'):
    """Require a completed finite inward trajectory covering every requested radius.

    SciPy dense output extrapolates outside its domain.  Such extrapolation can
    manufacture Taylor-like convergence even for a singular solution, so a
    successful status and actual coverage are both required before sampling.
    """
    if not getattr(sol, 'success', False) or getattr(sol, 'status', None) != 0:
        raise IntegrationError('%s did not complete: %s' % (context, getattr(sol, 'message', 'no solver status')))
    t = np.asarray(sol.t, dtype=float)
    y = np.asarray(sol.y, dtype=float)
    if t.ndim != 1 or t.size < 2 or not np.all(np.isfinite(t)) or not np.all(np.diff(t) < 0):
        raise IntegrationError('%s has an invalid or non-finite inward integration mesh' % context)
    if y.shape != (4, t.size) or not np.all(np.isfinite(y)):
        raise IntegrationError('%s has an invalid or non-finite stored trajectory' % context)
    if not callable(getattr(sol, 'sol', None)):
        raise IntegrationError('%s has no dense output' % context)
    requested = np.asarray(radii, dtype=float)
    if not np.all(np.isfinite(requested)):
        raise IntegrationError('%s requested a non-finite sample radius' % context)
    if np.any(requested < t[-1]) or np.any(requested > t[0]):
        raise IntegrationError('%s does not cover requested radii; computed domain is [%.17g, %.17g]' %
                               (context, t[-1], t[0]))


def sample_inward(sol, radii, context='inward sample'):
    """Evaluate finite dense data strictly inside a validated computed domain."""
    validate_inward(sol, radii, context)
    try:
        values = np.asarray(sol.sol(radii), dtype=float)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        raise IntegrationError('%s could not evaluate dense output: %s' % (context, exc)) from exc
    expected_shape = (4,) + np.asarray(radii).shape
    if values.shape != expected_shape or not np.all(np.isfinite(values)):
        raise IntegrationError('%s returned invalid or non-finite dense values' % context)
    return values


def area_product(bp, bm, vs):
    """Compute the provisional area product without accepting non-finite arithmetic."""
    try:
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            product = np.float64(bp) ** 2 * np.float64(bm) ** 2 / np.float64(vs) ** 2
    except FloatingPointError as exc:
        raise IntegrationError('area product has non-finite arithmetic: %s' % exc) from exc
    if not np.isfinite(product):
        raise IntegrationError('area product is non-finite')
    return product


def finite_spread(values):
    """A max-minus-min scientific comparison must inspect every input first."""
    values = tuple(values)
    ck.max_abs(values)
    spread = max(values) - min(values)
    ck.max_abs([spread])
    return spread


def _unresolved(reason, ests=None, rho=np.nan):
    estimates = tuple(ests) if ests is not None else (np.nan,) * (len(FITS) - 1)
    c = estimates[-1] if np.isfinite(estimates[-1]) else np.nan
    return dict(outcome='unresolved', c=c, u=np.inf, rho=rho, ests=estimates,
                launch=np.nan, solver=np.nan, product=None, reason=reason)


def model(a1, a2, Q1=1.0, Q2=1.0):
    V = lambda p: Q1 ** 2 * np.exp(a1 * p) + Q2 ** 2 * np.exp(a2 * p)
    dV = lambda p: a1 * Q1 ** 2 * np.exp(a1 * p) + a2 * Q2 ** 2 * np.exp(a2 * p)
    d2V = lambda p: a1 ** 2 * Q1 ** 2 * np.exp(a1 * p) + a2 ** 2 * Q2 ** 2 * np.exp(a2 * p)
    res = minimize_scalar(V, bounds=(-20, 20), method='bounded', options={'xatol': 1e-14})
    return V, dV, d2V, res.x, res.fun


def doubly_extremal(a1, a2):
    V, dV, d2V, ps, Vs = model(a1, a2)
    beta = brentq(lambda b: (RP + b) * (RM + b) - Vs, -RM + 1e-12, 100, xtol=1e-15)
    return ps, RP + beta, Vs


def horizon_data(a1, a2, bp, php):
    V, dV, d2V, ps, Vs = model(a1, a2)
    P1 = RP - RM
    bd = bp * (1 - V(php) / bp ** 2) / P1
    phd = dV(php) / (2 * bp ** 2 * P1)
    Sd = d2V(php) * phd / (2 * bp ** 2) - dV(php) * bd / bp ** 3
    phdd = (Sd - 2 * phd) / (2 * P1)          # from differentiating (P phi')' = S at r+
    bdd = -bp * phd ** 2
    return bd, phd, phdd, bdd


def rhs_factory(a1, a2):
    V, dV, d2V, ps, Vs = model(a1, a2)
    P = lambda r: (r - RP) * (r - RM)
    dP = lambda r: 2 * r - RP - RM
    return lambda r, y: [y[1], -y[0] * y[3] ** 2, y[3], (dV(y[2]) / (2 * y[0] ** 2) - dP(r) * y[3]) / P(r)]


def launch(a1, a2, bp, php, d, sign):
    """data at r = r+ + sign d from the Taylor expansion at the outer horizon"""
    bd, phd, phdd, bdd = horizon_data(a1, a2, bp, php)
    return [bp + sign * d * bd + d * d * bdd / 2, bd + sign * d * bdd, php + sign * d * phd + d * d * phdd / 2,
            phd + sign * d * phdd]


def outward(a1, a2, bp, php, d=1e-6, rtol=1e-12, atol=1e-14, rmax=1e5):
    def hit_zero(r, y):
        return y[0] - 1e-6
    hit_zero.terminal = True
    so = solve_ivp(rhs_factory(a1, a2), [RP + d, rmax], launch(a1, a2, bp, php, d, +1), method='DOP853', rtol=rtol,
                   atol=atol, events=hit_zero)
    return dict(flat=(so.status == 0 and so.y[1, -1] > 0), lambda_inf=so.y[1, -1], phi_inf=so.y[2, -1])


def inward(a1, a2, bp, php, d=1e-6, rtol=1e-12, atol=1e-14, rend=REND):
    try:
        sol = solve_ivp(rhs_factory(a1, a2), [RP - d, RM + rend], launch(a1, a2, bp, php, d, -1), method='DOP853',
                        rtol=rtol, atol=atol, dense_output=True)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        raise IntegrationError('inward solver failed: %s' % exc) from exc
    validate_inward(sol, [RP - d, RM + rend])
    return sol


def two_radius(sol, e1, e2):
    """x(e) = (r - r-) phi' = c + c1 e + ... ; the two-radius estimate (x1 e2 - x2 e1)/(e2 - e1) removes the O(e)
    term.  For a regular horizon x is a Taylor series and the error is O(e1 e2); for a singular one it involves
    fractional powers and logarithms.  Also returns b(r-) by linear extrapolation (meaningful only if regular)."""
    if not np.all(np.isfinite([e1, e2])) or not e1 > e2 > 0:
        raise IntegrationError('two-radius fit requires finite offsets e1 > e2 > 0')
    B1, _, _, p1 = sample_inward(sol, RM + e1, 'two-radius fit (coarse point)')
    B2, _, _, p2 = sample_inward(sol, RM + e2, 'two-radius fit (fine point)')
    x1, x2 = e1 * p1, e2 * p2
    estimate = (x1 * e2 - x2 * e1) / (e2 - e1)
    bminus = B2 - (B1 - B2) * e2 / (e1 - e2)
    if not np.all(np.isfinite([estimate, bminus])):
        raise IntegrationError('two-radius fit produced non-finite estimates')
    return estimate, bminus


def run(a1, a2, bp, php, d=1e-6, rtol=1e-12, atol=1e-14, rmax=1e5, fit=(1e-5, 1e-6), exterior=True):
    V, dV, d2V, ps, Vs = model(a1, a2)
    bd = horizon_data(a1, a2, bp, php)[0]
    out = dict(bprime=bd, physical=bd > 0)
    if exterior:
        out.update(outward(a1, a2, bp, php, d, rtol, atol, rmax))
    si = inward(a1, a2, bp, php, d, rtol, atol)
    validate_inward(si, [RM + REND], 'run inward integration')
    out['plateau'], Bm = two_radius(si, *fit)
    out['bminus'] = Bm
    out['product'] = area_product(bp, Bm, Vs)      # a horizon quantity only if the inner horizon is regular
    return out


def diagnose(a1, a2, bp, php, resolution=1e-9, floor=1e-12):
    """Three-outcome diagnostic of c = lim (r - r-) phi' at r-.
    c_k: two-radius estimates from the nested pairs of FITS, at launch distance d = 1e-6 from r+; the finest one is
    repeated at d = 1e-7 and with tighter solver tolerances (rtol = 1e-13).  The successive differences D_k = c_{k+1} - c_k shrink by a factor rho per decade, with
    rho ~ 0.01 for a regular horizon (x = (r-r-) phi' is then a Taylor series, error O(e1 e2)) and larger for a
    singular one.  With rho the largest observed ratio, the tail beyond the finest estimate is D rho/(1 - rho), so
        c = c_fine + D rho/(1-rho)  +-  u,
        u = |D| rho/(1-rho) + |c_fine(d=1e-6) - c_fine(d=1e-7)| + |c_fine(rtol=1e-12) - c_fine(rtol=1e-13)| + floor,
    with floor = 1e-12.  Outcomes:
        'nonzero'     |c| > 5u: a resolved logarithmic term, the inner horizon is singular;
        'zero'        |c| <= 5u and u <= resolution: consistent with a regular inner horizon at that resolution;
        'unresolved'  otherwise, if the estimates do not converge (rho >= 0.9), or if any main/refined integration
                      fails, lacks coverage, or contains non-finite data; the reason is returned and no product is given."""
    V, dV, d2V, ps, Vs = model(a1, a2)
    stage = 'main integration'
    ests = None
    try:
        sol = inward(a1, a2, bp, php, d=1e-6)
        ests, Bs = zip(*[two_radius(sol, FITS[k], FITS[k + 1]) for k in range(len(FITS) - 1)])
        stage = 'launch-refined integration'
        fine_d7 = two_radius(inward(a1, a2, bp, php, d=1e-7), FITS[-2], FITS[-1])[0]
        stage = 'tolerance-refined integration'
        fine_tol = two_radius(inward(a1, a2, bp, php, d=1e-6, rtol=1e-13, atol=1e-15), FITS[-2], FITS[-1])[0]
        stage = 'fit extrapolation'
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            ck.max_abs([ests, Bs, fine_d7, fine_tol, resolution, floor])
            D = [ests[k + 1] - ests[k] for k in range(len(ests) - 1)]
            ck.max_abs(D)
            ratios = [abs(D[k + 1]) / abs(D[k]) if D[k] != 0 else (0.0 if D[k + 1] == 0 else np.inf)
                      for k in range(len(D) - 1)]
            rho = float(ck.max_abs(ratios))
            if rho >= 0.9:
                return _unresolved('nested fit estimates do not converge', ests, rho)
            tail = D[-1] * rho / (1 - rho)
            c = ests[-1] + tail
            launch_error = abs(ests[-1] - fine_d7)
            solver_error = abs(ests[-1] - fine_tol)
            u = abs(tail) + launch_error + solver_error + floor
            confidence_width = 5 * u
            ck.max_abs([tail, c, launch_error, solver_error, u, confidence_width])
            if abs(c) > confidence_width:
                outcome = 'nonzero'
            elif u <= resolution:
                outcome = 'zero'
            else:
                outcome = 'unresolved'
            stage = 'horizon product'
            product = area_product(bp, Bs[-1], Vs) if outcome == 'zero' else None
    except (IntegrationError, ValueError, FloatingPointError, OverflowError) as exc:
        return _unresolved('%s: %s' % (stage, exc), ests)
    result = dict(outcome=outcome, c=c, u=u, rho=rho, ests=ests, launch=launch_error,
                  solver=solver_error, product=product)
    if outcome == 'unresolved':
        result['reason'] = 'estimated uncertainty %.3g exceeds resolution %.3g' % (u, resolution)
    return result


def ceil1(x):
    """x rounded UP to one significant digit (a bound)"""
    e = int(np.floor(np.log10(x)))
    m = int(np.ceil(x / 10.0 ** e - 1e-9))
    if m == 10:
        m, e = 1, e + 1
    return m * 10.0 ** e


def fmt_c(dg):
    if dg['outcome'] == 'unresolved' and dg.get('reason'):
        return 'unresolved: %s' % dg['reason']
    if dg['outcome'] == 'zero':
        return 'c = 0 (|c| < %.0e)' % ceil1(float(ck.max_abs([abs(dg['c']) + 5 * dg['u'], 1e-12])))
    return 'c = %+.6e +- %.1e' % (dg['c'], dg['u'])


MODELS = [((2.0, -2.0), 'A1+A1'), ((1.0, -4.0), 'A1+A1'), ((4.0, -1.0), 'A1+A1'),
          ((2 * np.sqrt(3), -2 * np.sqrt(3)), 'A2'), ((4.0, -6.0), 'B2'), ((10 / np.sqrt(3), -6 * np.sqrt(3)), 'G2'),
          ((2.0, -6.0), 'no'), ((1.0, -12.0), 'no'), ((3.0, -8.0), 'no'), ((5.0, -12.0), 'no'),
          ((2 * np.sqrt(10), -2 * np.sqrt(10)), 'no'),
          ((2.0, -1.0), 'no'), ((1.0, -1.0), 'no'), ((3.0, -3.0), 'no')]
# Reference for the eps-scaling: the constant-scalar (doubly-extremal) Reissner-Nordstrom solution with
# b+ = KB_REF * b+^DE at the same coordinate r+-; any b+ > sqrt(V_*) is such a solution (after t -> t/lambda,
# r -> lambda r).  KB_REF = 2 keeps the exterior asymptotically flat for every model and every eps in EPS.
KB_REF = 2.0
EPS = (0.04, 0.02, 0.01, 0.005)


def nu_of(a1, a2):
    return (np.sqrt(1 - 2 * a1 * a2) - 1) / 2


def scaling(verbose=True):
    """c(eps) for phi+ = phi_* + eps at b+ = KB_REF b+^DE, eps -> eps/2; ratios 2,4,8 <-> orders 1,2,3."""
    res = {}
    for (a1, a2), toda in MODELS:
        ps, bde, Vs = doubly_extremal(a1, a2)
        dgs, ok = [], True
        for eps in EPS:
            dgs.append(diagnose(a1, a2, KB_REF * bde, ps + eps))
            o = outward(a1, a2, KB_REF * bde, ps + eps)
            ok = ok and horizon_data(a1, a2, KB_REF * bde, ps + eps)[0] > 0 and o['flat']
        outcomes = [g['outcome'] for g in dgs]
        vals = [g['c'] for g in dgs]
        ratios = ([vals[i] / vals[i + 1] for i in range(len(EPS) - 1)] if set(outcomes) == {'nonzero'} else None)
        res[(a1, a2)] = (vals, ratios, ok, outcomes)
        if verbose:
            print("%-6s (%7.3f,%8.3f) nu=%.3f  physical&flat=%s  outcomes=%s  c(eps=%s) = %s  ratios = %s" % (
                toda, a1, a2, nu_of(a1, a2), ok, '/'.join(sorted(set(outcomes))), ','.join('%g' % e for e in EPS),
                ['%+.4e' % v for v in vals], ['%.3f' % x for x in ratios] if ratios else '-'))
    return res


def physical_data(a1, a2, want=2):
    """Pick horizon data with b'(r+) > 0 AND an asymptotically flat exterior."""
    ps, bde, Vs = doubly_extremal(a1, a2)
    picks = []
    for dphi in (0.2, -0.2, 0.1, -0.1, 0.05, -0.05, 0.02, -0.02):
        for kb in (1.3, 1.1, 1.5, 2.0, 2.5):
            o = run(a1, a2, kb * bde, ps + dphi)
            if o['physical'] and o['flat']:
                picks.append((kb, dphi, o))
                break
        if len(picks) >= want:
            break
    return picks


def table_physical():
    rows = []
    for (a1, a2), toda in MODELS:
        for kb, dphi, o in physical_data(a1, a2):
            ps, bde, Vs = doubly_extremal(a1, a2)
            dg = diagnose(a1, a2, kb * bde, ps + dphi)
            rows.append(((a1, a2), toda, nu_of(a1, a2), kb, dphi, o, dg))
            print("%-6s (%7.3f,%8.3f) nu=%.3f  b+/bDE=%.2f dphi=%+.2f : b'(r+)=%+.3f lam=%.3f phi_inf=%+.4f | %-10s %s  "
                  "[nested: %s; rate 1/rho = %.0f] %s" % (
                      toda, a1, a2, nu_of(a1, a2), kb, dphi, o['bprime'], o['lambda_inf'], o['phi_inf'], dg['outcome'],
                      fmt_c(dg), ' '.join('%+.2e' % e for e in dg['ests']), 1 / dg['rho'] if dg['rho'] > 0 else np.inf,
                      'prod=%.10f' % dg['product'] if dg['product'] is not None else ''))
    return rows


def second_order(verbose=True):
    """Integer nu: compare plateau/eps^2 at b+ = KB_REF b+^DE with c2 = (-1)^nu V3/(8 V_*) int_{-1}^{1} P_nu^3 dz
    (the prediction is independent of b+, i.e. of the mass at fixed charges)."""
    res = {}
    for (a1, a2), toda in MODELS:
        nu = nu_of(a1, a2)
        if abs(nu - round(nu)) > 1e-9:
            continue
        nu = int(round(nu))
        V, dV, d2V, ps, Vs = model(a1, a2)
        d3V = a1 ** 3 * np.exp(a1 * ps) + a2 ** 3 * np.exp(a2 * ps)
        I3 = quad(lambda z: eval_legendre(nu, z) ** 3, -1, 1)[0]
        c2 = (-1) ** nu * d3V / (8 * Vs) * I3
        ps_, bde, _ = doubly_extremal(a1, a2)
        num = []
        for kb in (1.3, KB_REF):
            num += [run(a1, a2, kb * bde, ps + e, exterior=False)['plateau'] / e ** 2 for e in (0.01, 0.005)]
        c2_closed = (-1) ** nu * nu * (nu + 1) * (a1 + a2) * I3 / 4     # two-exponential model
        res[(a1, a2)] = (c2, num)
        if verbose:
            print("%-6s (%7.3f,%8.3f) nu=%d  V3/V*=%+9.3f  int P^3=%.5f  c2 predicted=%+.5f (closed form %+.5f)  "
                  "plateau/eps^2 (b+/bDE=1.3: eps=.01,.005; b+/bDE=%.1f: eps=.01,.005) = %s" % (
                      toda, a1, a2, nu, d3V / Vs + 0.0, abs(I3) if abs(I3) < 1e-12 else I3, c2 + 0.0, c2_closed + 0.0,
                      KB_REF, ['%+.5f' % x for x in num]))
    return res


def kasner(verbose=True):
    """Singular cases: d ln b / d ln(r - r-) -> p with p(1 - p) = plateau^2 (Kasner exponents)."""
    out = []
    for (a1, a2, kb, dphi) in [(2.0, -1.0, 1.3, 0.2), (1.0, -1.0, 1.3, 0.2), (2.0, -6.0, 1.3, 0.2), (3.0, -3.0, 1.5, 0.2)]:
        ps, bde, Vs = doubly_extremal(a1, a2)
        bp, php = kb * bde, ps + dphi
        o = run(a1, a2, bp, php)
        sol = inward(a1, a2, bp, php)
        dg = diagnose(a1, a2, bp, php)
        c = dg['c']
        p_pred = (1 - np.sqrt(1 - 4 * c * c)) / 2
        slopes = []
        for e in (1e-6, 1e-7, 1e-8):
            B, B1, ph, ph1 = sample_inward(sol, RM + e, 'Kasner slope')
            slopes.append(e * B1 / B)
        out.append((a1, a2, c, p_pred, slopes, dg['outcome']))
        if verbose:
            print("(%5.2f,%6.2f) physical=%s flat=%s  plateau c=%+.5e  p from p(1-p)=c^2: %.5e   d ln b/d ln(r-r-) at "
                  "1e-6,1e-7,1e-8: %s" % (a1, a2, o['physical'], o['flat'], c, p_pred, ['%.5e' % x for x in slopes]))
    return out


def fit_floor():
    """Nested two-radius estimates: for a regular (Toda) horizon they fall by a factor ~100 per decade (Taylor
    series, error O(e1 e2)); for a singular one they converge to c != 0 at a slower, non-Taylor rate."""
    out = []
    cases = [(m, t) for m, t in MODELS if t != 'no'] + [((2.0, -6.0), 'no'), ((2.0, -1.0), 'no')]
    for (a1, a2), toda in cases:
        for kb, dphi, o in physical_data(a1, a2, want=1):
            ps, bde, Vs = doubly_extremal(a1, a2)
            dg = diagnose(a1, a2, kb * bde, ps + dphi)
            out.append(((a1, a2), toda, dg))
            print("%-6s (%7.3f,%8.3f) b+/bDE=%.2f dphi=%+.2f : estimates at (1e-4,1e-5) ... (1e-7,1e-8) = %s ; "
                  "differences shrink by 1/rho = %.1f per decade -> %s, %s" % (
                      toda, a1, a2, kb, dphi, ' '.join('%+.4e' % e for e in dg['ests']), 1 / dg['rho'], dg['outcome'],
                      fmt_c(dg)))
    return out


def small_signals(verbose=True):
    """Small but nonzero logarithmic coefficients.  An absolute threshold |c| < 1e-6 would call
    these regular.  The diagnostic resolves them, and they agree with the perturbative predictions
    c = c2 eps^2 (c2 = -24/35 for (2,-6)) and c = c3 eps^3 (c3 = 24 (s^2-4)/1001 = 504/1001 for (3,-8))."""
    out = []
    for (a1, a2, eps, pred) in ((2.0, -6.0, 0.001, -24 / 35 * 0.001 ** 2), (3.0, -8.0, 0.005, 504 / 1001 * 0.005 ** 3)):
        ps, bde, Vs = doubly_extremal(a1, a2)
        dg = diagnose(a1, a2, 2.0 * bde, ps + eps)
        out.append((a1, a2, eps, pred, dg))
        if verbose:
            print("(%g,%g) eps = %g, b+ = 2 b+^DE: %s, %s ; perturbative prediction %+.4e" % (
                a1, a2, eps, dg['outcome'], fmt_c(dg), pred))
    return out


def nu3_scan(verbose=True):
    """nu = 3 family alpha1 alpha2 = -24: third-order coefficient c3 = lim c/eps^3 (Richardson from eps = 0.01, 0.005
    at b+ = KB_REF b+^DE, fit radii (1e-6, 1e-7)), against the exact c3 = 24 (s^2 - 4)/1001, s = alpha1 + alpha2
    (c3_exact.py).  c2 = 0 on the whole family (odd nu); c3 vanishes only at the B2 couplings (4,-6) and (6,-4)."""
    out = []
    for a1 in (2.0, 2.5, 3.0, 3.5, 3.9, 4.0, 4.1, 4.5, np.sqrt(24.0), 5.5, 5.9, 6.0, 6.1, 6.5, 7.0, 8.0):
        a2 = -24.0 / a1
        ps, bde, Vs = doubly_extremal(a1, a2)
        x = []
        for e in (0.01, 0.005):
            o = run(a1, a2, KB_REF * bde, ps + e, fit=(1e-6, 1e-7), exterior=False)
            x.append(o['plateau'] / e ** 3)
        c3 = 2 * x[1] - x[0]
        exact = 24 * ((a1 + a2) ** 2 - 4) / 1001
        out.append((a1, a2, c3, exact))
        if verbose:
            print("   (%6.3f,%8.3f)  c3 ~ %+.4f   exact 24(s^2-4)/1001 = %+.4f   (c/eps^3 at eps=.01,.005: %+.4f, %+.4f)"
                  % (a1, a2, c3, exact, x[0], x[1]))
    return out


SYM_NU3 = np.sqrt(24.0)       # alpha1 = -alpha2 = sqrt(24): nu = 3


def alpha_of_nu(nu):
    return np.sqrt(((2 * nu + 1) ** 2 - 1) / 2)


def c_fine(a1, a2, bp, php):
    """finest two-radius estimate, (1e-7, 1e-8), for root finding"""
    return two_radius(inward(a1, a2, bp, php), FITS[-2], FITS[-1])[0]


def tuned_alpha(eps, kb=2.5, xtol=1e-13):
    """symmetric couplings alpha1 = -alpha2 = alpha: the alpha below sqrt(24) at which c = 0 for phi+ - phi_* = eps"""
    def f(al):
        ps, bde, Vs = doubly_extremal(al, -al)
        return c_fine(al, -al, kb * bde, ps + eps)
    lo = alpha_of_nu(3 - 3 * (96 / 1001) * eps ** 2 - 1e-4)
    return brentq(f, lo, SYM_NU3 - 1e-9, xtol=xtol)


def tau_symmetry(verbose=True):
    """The radial equations are autonomous in tau, d tau = dr/(a^2 b^2).  At fixed charges and fixed coordinate
    radii r_pm a shift of tau maps solutions to solutions; it rescales b+ -> lambda b+ and b- -> b-/lambda and keeps
    phi_pm and c.  Hence c depends only on (phi+, coupling) and, for regular inner horizons, b+ b- is constant along each
    orbit phi+ = const.  Checked here on a singular and a regular (A2) example."""
    out = []
    for (a1, a2), eps in (((2.0, -6.0), 0.2), ((2.0, -6.0), 0.01), ((2 * np.sqrt(3), -2 * np.sqrt(3)), 0.2)):
        ps, bde, Vs = doubly_extremal(a1, a2)
        dgs = [diagnose(a1, a2, kb * bde, ps + eps) for kb in (1.3, 2.0, 2.5)]
        out.append(((a1, a2), eps, dgs))
        if verbose:
            print("   (%.3f,%.3f) eps = %g, b+/bDE = 1.3, 2.0, 2.5: c = %s ; products %s" % (
                a1, a2, eps, ', '.join('%+.10e' % g['c'] for g in dgs),
                ', '.join('%.12f' % g['product'] if g['product'] is not None else '-' for g in dgs)))
    return out


def tuned_branch(verbose=True):
    """Davydov's 'second class' (arXiv:1711.04198) in the symmetric model: regular inner horizons at non-integer nu.
    (i)   b+ = 2.5 b+^DE, eps = 0.2: shoot in alpha for c = 0 (the finite-amplitude example);
    (ii)  at that fixed alpha (same theory, same charges Q1 = Q2 = 1) the whole tau-orbit phi+ = phi_* + 0.2 is regular
          (tau_symmetry): b+/bDE = 3, 4 at the same eps, with b+^2 b-^2/V_*^2 constant along it (by the symmetry) but
          not 1; the orbit has asymptotically flat exteriors only for b+/bDE >~ 2.4, where b-/b+ <~ 0.11, so it never
          approaches extremality;
    (iii) the curve of tuned couplings nu(eps) through nu = 3: c = eps sin(pi nu)/pi + c3 eps^3 + ..., so
          3 - nu = -c3 eps^2 + O(eps^4), with c3 = -96/1001 at s = 0 (c3_exact.py);
    (iv)  at nu = 3 exactly the same data are singular, c ~ c3 eps^3."""
    res = {}
    al = tuned_alpha(0.2)
    nu = nu_of(al, -al)
    ps, bde, Vs = doubly_extremal(al, -al)
    rows = []
    for kb in (2.5, 3.0, 4.0):
        dg = diagnose(al, -al, kb * bde, ps + 0.2)
        o = outward(al, -al, kb * bde, ps + 0.2)
        phys = horizon_data(al, -al, kb * bde, ps + 0.2)[0] > 0
        ratio = sample_inward(inward(al, -al, kb * bde, ps + 0.2), RM + FITS[-1], 'tuned-branch radius ratio')[0] / (kb * bde)
        rows.append((kb, 0.2, dg, o, phys, ratio))
        if verbose:
            print("   alpha = %.12f (nu = %.10f), b+/bDE = %.1f, eps = 0.2: physical=%s flat=%s phi_inf=%.8f b-/b+=%.4f | %s %s "
                  "[nested: %s; the last is the shooting target at b+/bDE = 2.5]  product = %s" % (
                      al, nu, kb, phys, o['flat'], o['phi_inf'], ratio, dg['outcome'], fmt_c(dg),
                      ' '.join('%+.2e' % e for e in dg['ests']),
                      '%.11f' % dg['product'] if dg['product'] is not None else '-'))
    res['alpha'], res['nu'], res['rows'] = al, nu, rows
    ends = []
    for kb in (2.3, 2.4):
        o = outward(al, -al, kb * bde, ps + 0.2)
        ratio = sample_inward(inward(al, -al, kb * bde, ps + 0.2), RM + FITS[-1], 'tuned-branch endpoint ratio')[0] / (kb * bde)
        ends.append((kb, o['flat'], ratio, o['phi_inf']))
        if verbose:
            print("   end of the physical orbit: b+/bDE = %.1f: asymptotically flat = %s, b-/b+ = %.4f" % (kb, o['flat'], ratio))
    res['ends'] = ends
    br = []
    for eps in (0.05, 0.1, 0.2):
        a_e = tuned_alpha(eps)
        d_e = 3 - nu_of(a_e, -a_e)
        br.append((eps, a_e, d_e))
        if verbose:
            print("   tuned couplings: eps = %.2f -> alpha = %.12f, 3 - nu = %.6e, (3 - nu)/eps^2 = %.6f   [-c3 = 96/1001 = %.6f]"
                  % (eps, a_e, d_e, d_e / eps ** 2, 96 / 1001))
    res['branch'] = br
    ps3, bde3, _ = doubly_extremal(SYM_NU3, -SYM_NU3)
    dg3 = diagnose(SYM_NU3, -SYM_NU3, 2.5 * bde3, ps3 + 0.2)
    res['nu3'] = dg3
    if verbose:
        print("   at nu = 3 exactly (alpha = sqrt 24), b+/bDE = 2.5, eps = 0.2: %s, %s ; c3 eps^3 = %+.4e"
              % (dg3['outcome'], fmt_c(dg3), -96 / 1001 * 0.2 ** 3))
    return res


if __name__ == '__main__':
    print("--- physical outer-horizon data: b'(r+) > 0 and asymptotically flat exterior; three-outcome diagnostic ---")
    rows = table_physical()
    for (a12, toda, nu, kb, dphi, o, dg) in rows:
        lab = '(%.3f,%.3f) b+/bDE=%.2f dphi=%+.2f' % (a12[0], a12[1], kb, dphi)
        ck.true('table %s: physical data with an asymptotically flat exterior' % lab, o['physical'] and o['flat'])
        if toda != 'no':
            ck.true('table %s (%s): inner horizon regular (outcome zero at resolution 1e-9)' % (lab, toda), dg['outcome'] == 'zero')
            ck.small('table %s (%s): b+^2 b-^2/V_*^2 = 1' % (lab, toda), dg['product'] - 1 if dg['product'] is not None else np.inf, 1e-8)
        else:
            ck.true('table %s (non-Toda, nu = %.3f): inner horizon singular (outcome nonzero)' % (lab, nu), dg['outcome'] == 'nonzero')
    reg = [r_ for r_ in rows if r_[1] != 'no' and r_[6]['product'] is not None]
    if reg:
        rates = [1 / r_[6]['rho'] for r_ in reg]
        ck.max_abs(rates)
        print('Toda rows: max |b+^2 b-^2/V*^2 - 1| = %.1e ;  largest bound |c| + 5u = %.1e ;  rates 1/rho = %.0f-%.0f' % (
            ck.max_abs(r_[6]['product'] - 1 for r_ in reg), ck.max_abs(abs(r_[6]['c']) + 5 * r_[6]['u'] for r_ in reg),
            min(rates), max(rates)))
    sing = [r_ for r_ in rows if r_[1] == 'no']
    signal_ratios = [abs(r_[6]['c']) / r_[6]['u'] for r_ in sing]
    rates = [1 / r_[6]['rho'] for r_ in sing]
    ck.max_abs([signal_ratios, rates])
    print('non-Toda rows: smallest |c|/u = %.1e ;  rates 1/rho = %.0f-%.0f' % (
        min(signal_ratios), min(rates), max(rates)))
    print('\n--- order of the obstruction: c vs eps = phi+ - phi_* at b+ = %.1f b+^DE (ratios 2,4,8 <-> eps^1,2,3) ---' % KB_REF)
    sc = scaling()
    for (a1, a2), toda in MODELS:
        vals, ratios, ok, outcomes = sc[(a1, a2)]
        nu = nu_of(a1, a2)
        lab = '(%.3f,%.3f)' % (a1, a2)
        ck.true('scaling %s: physical data for all eps' % lab, ok)
        if toda != 'no':
            ck.true('scaling %s (%s): outcome zero for every eps' % (lab, toda), set(outcomes) == {'zero'})
            continue
        ck.true('scaling %s: outcome nonzero for every eps' % lab, set(outcomes) == {'nonzero'})
        if abs(nu - round(nu)) > 1e-9:
            order = 1
        else:
            order = 2 if (round(nu) % 2 == 0 and abs(a1 + a2) > 1e-9) else 3
        ck.true('scaling %s (nu = %.3f): c ~ eps^%d' % (lab, nu, order),
                ratios is not None and all(abs(x - 2 ** order) < 0.25 * order for x in ratios),
                'ratios %s' % (['%.3f' % x for x in ratios] if ratios else None))
    print("\n--- second-order obstruction: numerics vs c2 = (-1)^nu V3/(8 V*) int P_nu^3 ---")
    so2 = second_order()
    for (a1, a2), (c2, num) in so2.items():
        if abs(c2) > 1e-3:
            ext = [2 * num[1] - num[0], 2 * num[3] - num[2]]       # linear extrapolation of c/eps^2 to eps = 0
            ck.close('second order (%.0f,%.0f): c/eps^2 extrapolated to eps = 0, for b+/bDE = 1.3 and 2, equals c2 = %.5f'
                     % (a1, a2, c2), ck.max_abs(x - c2 for x in ext), 0, 2e-3 * abs(c2))
    print('\n--- singular inner horizons are Kasner: b ~ (r-r-)^p with p(1-p) = c^2 ---')
    for (a1, a2, c, p_pred, slopes, outcome) in kasner():
        ck.close('Kasner (%g,%g): d ln b/d ln(r - r-) at r - r- = 1e-8 approaches p' % (a1, a2), slopes[-1], p_pred, 2e-3, rel=True)
    print('\n--- nested two-radius estimates: Taylor-like (regular) versus non-Taylor (singular) convergence ---')
    for (a12, toda, dg) in fit_floor():
        if toda != 'no':
            ck.true('fit (%.3f,%.3f) %s: estimates fall by >= 50 per decade (O(e1 e2) error of a regular horizon)'
                    % (a12[0], a12[1], toda), dg['rho'] < 0.02, '1/rho = %.1f' % (1 / dg['rho']))
        else:
            ck.true('fit (%.3f,%.3f): singular, converges to c != 0 at a non-Taylor rate' % a12,
                    dg['outcome'] == 'nonzero' and dg['rho'] > 0.02, '1/rho = %.1f' % (1 / dg['rho']))
    print('\n--- small signals: tiny nonzero c are resolved, not classified as regular ---')
    for (a1, a2, eps, pred, dg) in small_signals():
        ck.true('small signal (%g,%g) eps = %g: outcome nonzero' % (a1, a2, eps), dg['outcome'] == 'nonzero')
        ck.close('small signal (%g,%g) eps = %g: c agrees with the perturbative prediction' % (a1, a2, eps), dg['c'], pred,
                 0.01, rel=True)
    print('\n--- nu = 3 family (alpha1 alpha2 = -24): third-order obstruction coefficient ---')
    for (a1, a2, c3, exact) in nu3_scan():
        ck.close('nu = 3 scan (%.3f,%.3f): numerical c3 = 24 (s^2 - 4)/1001' % (a1, a2), c3, exact, 2e-3)
    print('\n--- tau-translation symmetry: c depends only on phi+; b+ b- is constant along an orbit phi+ = const ---')
    for (a12, eps, dgs) in tau_symmetry():
        cs = [g['c'] for g in dgs]
        spread = finite_spread(cs)
        uncertainty = ck.max_abs(g['u'] for g in dgs)
        ck.true('tau symmetry (%.3f,%.3f) eps = %g: c is the same at b+/bDE = 1.3, 2, 2.5 (within 3u)' % (a12[0], a12[1], eps),
                spread <= 3 * uncertainty, 'spread %.1e' % spread)
        if all(g['product'] is not None for g in dgs):
            ck.small('tau symmetry (%.3f,%.3f) eps = %g: b+^2 b-^2/V_*^2 the same along the orbit' % (a12[0], a12[1], eps),
                     finite_spread(g['product'] for g in dgs), 1e-9)
    print('\n--- tuned branches at non-integer nu (Davydov): regular inner horizons outside the Toda list ---')
    tb = tuned_branch()
    ck.close('tuned branch: alpha reproduces the reference value 4.893760362138778', tb['alpha'], 4.893760362138778, 1e-8)
    prods = []
    for kb, eps, dg, o, phys, ratio in tb['rows']:
        ck.true('tuned branch b+/bDE = %.1f, eps = 0.2: physical, asymptotically flat, inner horizon regular (outcome zero)' % kb,
                phys and o['flat'] and dg['outcome'] == 'zero')
        if dg['product'] is not None:
            prods.append(dg['product'])
    if len(prods) == 3:
        ck.small('tuned branch: b+^2 b-^2/V_*^2 is the same at b+/bDE = 2.5, 3, 4 (tau symmetry)', finite_spread(prods), 1e-9)
        ck.large('tuned branch: and it differs from 1', prods[0] - 1, 1e-5)
    ck.true('tuned branch: phi_inf differs along the orbit (a one-parameter family at fixed charges)',
            abs(tb['rows'][0][3]['phi_inf'] - tb['rows'][2][3]['phi_inf']) > 0.1)
    (k23, f23, r23, p23), (k24, f24, r24, p24) = tb['ends']
    ck.true('tuned branch: the exterior is asymptotically flat at b+/bDE = 2.4 but not at 2.3, where b-/b+ < 0.12: '
            'the physical orbit never approaches extremality', f24 and not f23 and r23 < 0.12, 'b-/b+ = %.4f, %.4f' % (r23, r24))
    d = [x[2] / x[0] ** 2 for x in tb['branch']]
    ck.close('tuned couplings bifurcate from nu = 3: (3 - nu)/eps^2 -> 96/1001 (eps = 0.05)', d[0], 96 / 1001, 0.02, rel=True)
    ck.true('tuned couplings: (3 - nu)/eps^2 approaches 96/1001 monotonically as eps -> 0',
            abs(d[0] - 96 / 1001) < abs(d[1] - 96 / 1001) < abs(d[2] - 96 / 1001))
    ck.true('at nu = 3 exactly the same data are singular', tb['nu3']['outcome'] == 'nonzero')
    ck.finish()
