r"""Independent two-horizon collocation for the symmetric dilaton model.

No inward-shooting code or numerical shooting data are imported.  With
P=(r-r+)(r-r-), W=2 cosh(alpha phi), the evolved variables are
(b, b', phi, J=P phi'); their equations are b''=-b phi'^2 and
J'=alpha sinh(alpha phi)/b^2.  scipy.solve_bvp solves all collocation
intervals simultaneously, from an analytic RN plus Legendre-mode seed.

A Taylor series at either simple horizon uses P=d*s+s^2.  If b=sum b_n s^n
and phi=sum q_n s^n, then
 d(n+1)^2 q_(n+1)+n(n+1)q_n=[alpha sinh(alpha phi)/b^2]_n,
 (n+2)(n+1)b_(n+2)=-[b(phi')^2]_n.
These regular expansions are matched at BOTH ends.  Outer b' satisfies the
Hamiltonian constraint.  Inner b' is a free unknown, leaving the inner
constraint as an independent check, not a redundant boundary condition.
Four field equations and four parameters (alpha,b-,b'-,phi-) require eight
matching conditions. At fixed alpha the fourth physical unknown is phi+.

An analytic RN Mobius coordinate resolves the inner horizon without loss
from subtracting nearby radial coordinates. The radial gauge remains the
same as in the paper; this coordinate is only a collocation mesh choice.
"""
from dataclasses import dataclass
import numpy as np
from numpy.polynomial import polynomial as poly
from numpy.polynomial.legendre import Legendre
from scipy.integrate import solve_bvp, solve_ivp

RP = 1.0
RM = 0.35
DELTA = RP - RM
BDE = (DELTA + np.sqrt(DELTA**2 + 8.0)) / 2.0


class BVPError(RuntimeError):
    """A collocation or exterior solve did not produce a usable solution."""


def _exp(q, n):
    result = np.zeros(n + 1)
    result[0] = np.exp(q[0])
    for m in range(1, n + 1):
        result[m] = sum(k*q[k]*result[m-k] for k in range(1, m+1))/m
    return result


def _inverse(q, n):
    result = np.zeros(n + 1)
    result[0] = 1.0/q[0]
    for m in range(1, n + 1):
        result[m] = -sum(q[k]*result[m-k] for k in range(1, m+1))/q[0]
    return result


def taylor_coefficients(b0, b1, phi0, alpha, d, order):
    """Power-series coefficients (not derivatives), computed by recurrence."""
    b, phi = np.zeros(order+2), np.zeros(order+2)
    b[:2], phi[0] = (b0, b1), phi0
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        for n in range(order):
            inv = _inverse(b, n)
            source = alpha*(_exp(alpha*phi, n)-_exp(-alpha*phi, n))/2
            fn = np.convolve(source, np.convolve(inv, inv))[:n+1][n]
            phi[n+1] = (fn-n*(n+1)*phi[n])/(d*(n+1)**2)
            if n+2 <= order:
                dp = np.arange(1, n+2)*phi[1:n+2]
                b[n+2] = -np.convolve(b[:n+1], np.convolve(dp, dp))[:n+1][n]/((n+2)*(n+1))
    return b[:order+1], phi[:order+1]


def taylor_fields(b0, b1, phi0, alpha, d, offset, order):
    """Return b,b',phi,phi' at a signed offset from a horizon."""
    b, phi = taylor_coefficients(b0, b1, phi0, alpha, d, order)
    return np.asarray([poly.polyval(offset, b), poly.polyval(offset, poly.polyder(b)),
                       poly.polyval(offset, phi), poly.polyval(offset, poly.polyder(phi))])


def outer_slope(bp, phip, alpha):
    return (bp-2*np.cosh(alpha*phip)/bp)/DELTA


@dataclass
class BranchSolution:
    collocation: object
    bp: float
    phip: float
    alpha: float
    bm: float
    bdm: float
    phim: float
    cutoff: float
    order: int
    tolerance: float
    initial_nodes: int
    degree: int
    fixed_alpha: bool
    boundary_residual: float

    @property
    def bdp(self):
        return outer_slope(self.bp, self.phip, self.alpha)

    @property
    def product(self):
        return self.bp**2*self.bm**2/4.0

    def radius(self, z):
        z = np.asarray(z)
        bmref = 2.0/self.bp
        return RM+DELTA*bmref*(z+1)/(self.bp*(1-z)+bmref*(z+1))

    def coordinate(self, radius):
        radius = np.asarray(radius)
        bmref = 2.0/self.bp
        background = bmref+(self.bp-bmref)*(radius-RM)/DELTA
        return 1-2*bmref*(RP-radius)/(DELTA*background)

    def radial_jacobian(self, z):
        bmref = 2.0/self.bp
        denominator = self.bp*(1-z)+bmref*(z+1)
        return 2*DELTA*self.bp*bmref/denominator**2

    def p_of_coordinate(self, z):
        bmref = 2.0/self.bp
        denominator = self.bp*(1-z)+bmref*(z+1)
        return -DELTA**2*self.bp*bmref*(1-z*z)/denominator**2

    def sample(self, radius):
        """Physical fields on [r-,r+], using the endpoint series where needed.

        No dense-output extrapolation is allowed. Endpoint series cover only
        the omitted cutoff intervals and are part of the stated approximation.
        """
        scalar = np.ndim(radius) == 0
        r = np.atleast_1d(np.asarray(radius, dtype=float))
        if r.ndim != 1 or not np.all(np.isfinite(r)) or np.any(r < RM) or np.any(r > RP):
            raise BVPError('requested profile lies outside the two horizons')
        values = np.empty((4, len(r)))
        inner, outer = r < RM+self.cutoff, r > RP-self.cutoff
        middle = ~(inner | outer)
        if np.any(inner):
            values[:, inner] = taylor_fields(self.bm, self.bdm, self.phim, self.alpha,
                                             -DELTA, r[inner]-RM, self.order)
        if np.any(outer):
            values[:, outer] = taylor_fields(self.bp, self.bdp, self.phip, self.alpha,
                                             DELTA, r[outer]-RP, self.order)
        if np.any(middle):
            z = self.coordinate(r[middle])
            # Roundoff in the analytic coordinate map can move a cutoff by one
            # ulp. Clip only that map, never a requested physical radius.
            lower, upper = self.collocation.x[0], self.collocation.x[-1]
            slack = 16*np.finfo(float).eps*max(1., abs(lower), abs(upper))
            if np.any(z < lower-slack) or np.any(z > upper+slack):
                raise BVPError('requested profile lies outside the computed collocation interval')
            z = np.clip(z, lower, upper)
            values[:, middle] = self.collocation.sol(z)
            values[3, middle] /= self.p_of_coordinate(z)
        if not np.all(np.isfinite(values)) or np.any(values[0] <= 0):
            raise BVPError('non-finite fields or a nonpositive area radius')
        return values[:, 0] if scalar else values

    def diagnostics(self, grid_size=1601):
        # Irrational offset avoids reusing collocation nodes as test points.
        u = (np.arange(grid_size)+np.sqrt(2)/3)/grid_size
        z = self.collocation.x[0]+u*(self.collocation.x[-1]-self.collocation.x[0])
        r = self.radius(z)
        y = self.collocation.sol(z)
        yr = self.collocation.sol(z, 1)/self.radial_jacobian(z)
        p = self.p_of_coordinate(z)
        dp = 2*r-RP-RM
        b, bd, phi, flux = y
        pd = flux/p
        source = self.alpha*np.sinh(self.alpha*phi)/b**2
        rhs = np.asarray([bd, -b*pd**2, pd, source])
        ode = (yr-rhs)/np.maximum(1, np.abs(rhs))
        w = 2*np.cosh(self.alpha*phi)
        h = -1+dp*bd/b-p*bd**2/b**2+w/b**2-p*pd**2
        bdd = yr[1]
        aa = p/b**2
        ad = dp/b**2-2*p*bd/b**3
        add = 2/b**2-4*dp*bd/b**3-2*p*bdd/b**3+6*p*bd**2/b**4
        ricci = -add-4*(aa*bdd+ad*bd)/b+2*(1-aa*bd**2)/b**2
        ricci_target = 2*aa*pd**2
        kretschmann = add**2+2*(ad*bd/b)**2+2*((2*aa*bdd+ad*bd)/b)**2+4*(1-aa*bd**2)**2/b**4
        endpoint_h = [-1-DELTA*self.bdm/self.bm+2*np.cosh(self.alpha*self.phim)/self.bm**2,
                      -1+DELTA*self.bdp/self.bp+2*np.cosh(self.alpha*self.phip)/self.bp**2]
        endpoint_k = []
        for bh, ph in ((self.bm, self.phim), (self.bp, self.phip)):
            wh = 2*np.cosh(self.alpha*ph)/bh**2
            endpoint_k.append((12-24*wh+20*wh**2)/bh**4)
        arrays = [y, yr, ode, h, ricci, kretschmann, endpoint_h, endpoint_k]
        if not all(np.all(np.isfinite(a)) for a in arrays) or np.min(b) <= 0:
            raise BVPError('non-finite independent diagnostics')
        return dict(alpha=float(self.alpha), nu=float((np.sqrt(1+2*self.alpha**2)-1)/2),
                    b_plus=float(self.bp), b_minus=float(self.bm),
                    phi_plus=float(self.phip), phi_minus=float(self.phim),
                    b_prime_plus=float(self.bdp), b_prime_minus=float(self.bdm),
                    scalar_prime_minus=float(self.alpha*np.sinh(self.alpha*self.phim)/(-DELTA*self.bm**2)),
                    scalar_prime_plus=float(self.alpha*np.sinh(self.alpha*self.phip)/(DELTA*self.bp**2)),
                    normalized_product=float(self.product), product_deficit=float(1-self.product),
                    entropy_gap_lower_over_Sstar=float(1-self.product),
                    entropy_gap_actual_over_Sstar=float((self.bp**2-self.bm**2)/2),
                    collocation_nodes=int(self.collocation.x.size),
                    collocation_max_rms=float(np.max(self.collocation.rms_residuals)),
                    boundary_max_abs=float(self.boundary_residual),
                    off_grid_scaled_ode_max=float(np.max(np.abs(ode))),
                    off_grid_ode_by_field=np.max(np.abs(ode), axis=1).tolist(),
                    hamiltonian_max_abs=float(np.max(np.abs(h))),
                    hamiltonian_constant_max_abs=float(np.max(np.abs(b*b*h))),
                    hamiltonian_constant_spread=float(np.ptp(b*b*h)),
                    horizon_hamiltonian=endpoint_h,
                    ricci_scaled_equation_error=float(np.max(np.abs(ricci-ricci_target)/np.maximum(1, np.abs(ricci_target)))),
                    kretschmann_horizons=endpoint_k,
                    kretschmann_interior_max=float(np.max(kretschmann)),
                    minimum_b=float(np.min(b)),
                    cutoff=self.cutoff, series_order=self.order, tolerance=self.tolerance,
                    initial_nodes=self.initial_nodes, legendre_seed_degree=self.degree,
                    fixed_alpha=self.fixed_alpha, success=True, status=int(self.collocation.status))


def solve_branch(kb=2.5, amplitude=0.2, alpha=None, cutoff=1e-4, order=8,
                 tolerance=1e-10, initial_nodes=301, degree=3, bp=None,
                 max_nodes=40000, seed_solution=None):
    """Solve afresh from analytic data; alpha=None tunes alpha at fixed phi+.

    With alpha supplied, it is held EXACTLY fixed and phi+ is solved instead.
    The amplitude then supplies only the initial guess, not a boundary datum.
    Optional seed_solution continues a previous BVP; no IVP result is accepted.
    """
    bp = kb*BDE if bp is None else float(bp)
    if (not np.all(np.isfinite([bp, amplitude, cutoff, tolerance])) or bp <= 0
            or not 0 < cutoff < DELTA/4 or not 2 <= order <= 12
            or not 1e-12 <= tolerance <= 1e-3 or initial_nodes < 20
            or degree < 1 or degree != int(degree) or amplitude == 0
            or (alpha is not None and (not np.isfinite(alpha) or alpha <= 0))):
        raise ValueError('invalid two-horizon collocation settings')
    bmref = 2/bp
    if abs(bp-bmref) < 1e-10:
        raise ValueError('the RN mesh seed requires distinct background horizon radii')

    def coordinate(r):
        background = bmref+(bp-bmref)*(r-RM)/DELTA
        return 1-2*bmref*(RP-r)/(DELTA*background)

    def geometry(z):
        denominator = bp*(1-z)+bmref*(z+1)
        rz = 2*DELTA*bp*bmref/denominator**2
        p = -DELTA**2*bp*bmref*(1-z*z)/denominator**2
        return rz, p

    def equations(z, y, parameters):
        coupling = parameters[0] if alpha is None else alpha
        rz, p = geometry(z)
        b, bd, phi, flux = y
        pd = flux/p
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            return rz*np.asarray([bd, -b*pd**2, pd, coupling*np.sinh(coupling*phi)/b**2])

    def boundary(ya, yb, parameters):
        coupling = parameters[0] if alpha is None else alpha
        phip = amplitude if alpha is None else parameters[0]
        bm, bdm, phim = parameters[1:]
        inner = taylor_fields(bm, bdm, phim, coupling, -DELTA, cutoff, order)
        outer = taylor_fields(bp, outer_slope(bp, phip, coupling), phip,
                              coupling, DELTA, -cutoff, order)
        inner[3] *= cutoff*(cutoff-DELTA)
        outer[3] *= cutoff*(cutoff-DELTA)
        return np.r_[ya-inner, yb-outer]

    z = np.linspace(coordinate(RM+cutoff), coordinate(RP-cutoff), initial_nodes)
    denominator = bp*(1-z)+bmref*(z+1)
    r = RM+DELTA*bmref*(z+1)/denominator
    b = bmref+(bp-bmref)*(r-RM)/DELTA
    polynomial = Legendre.basis(degree)
    phi = amplitude*polynomial(z)
    pd = amplitude*polynomial.deriv()(z)*2*bp*bmref/(DELTA*b**2)
    _, p = geometry(z)
    seed = np.asarray([b, np.full_like(z, (bp-bmref)/DELTA), phi, p*pd])
    parameters = [np.sqrt(2*degree*(degree+1)) if alpha is None else amplitude,
                  bmref, (bp-bmref)/DELTA, amplitude*(-1)**degree]
    if seed_solution is not None:
        if not isinstance(seed_solution, BranchSolution):
            raise TypeError('continuation seed must be a completed BranchSolution')
        old = seed_solution.sample(seed_solution.radius(z))
        old_jac = seed_solution.radial_jacobian(z)
        new_jac, p = geometry(z)
        delta_bp = bp-seed_solution.bp
        new_bm = seed_solution.bm*seed_solution.bp/bp
        delta_bm = new_bm-seed_solution.bm
        seed[0] = old[0]+(1+z)*delta_bp/2+(1-z)*delta_bm/2
        seed[1] = (old[1]*old_jac+(delta_bp-delta_bm)/2)/new_jac
        seed[2] = old[2]
        seed[3] = p*old[3]*old_jac/new_jac
        inner_jac, _ = geometry(np.asarray([-1.0]))
        new_bdm = (seed_solution.bdm*seed_solution.radial_jacobian(-1.0)+(delta_bp-delta_bm)/2)/inner_jac[0]
        parameters = [seed_solution.alpha if alpha is None else seed_solution.phip,
                      new_bm, new_bdm, seed_solution.phim]
    try:
        result = solve_bvp(equations, boundary, z, seed, p=np.asarray(parameters),
                           tol=tolerance, bc_tol=min(1e-12, tolerance/100), max_nodes=max_nodes)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        raise BVPError('collocation raised: '+str(exc)) from exc
    if (not result.success or result.status != 0 or not np.all(np.isfinite(result.y))
            or not np.all(np.isfinite(result.p)) or not np.all(np.isfinite(result.rms_residuals))):
        raise BVPError('collocation failed (status %s): %s; nodes=%s, residual=%s' %
                       (result.status, result.message, result.x.size, np.max(result.rms_residuals)))
    if (np.ndim(result.x) != 1 or result.x.size < 2
            or not np.all(np.isfinite(result.x)) or not np.all(np.diff(result.x) > 0)
            or result.x[0] > z[0] or result.x[-1] < z[-1]
            or np.shape(result.y) != (4, result.x.size) or np.shape(result.p) != (4,)
            or np.shape(result.rms_residuals) != (result.x.size-1,)
            or not callable(result.sol)):
        raise BVPError('collocation returned an invalid mesh, shape, or incomplete interval')
    coupling = float(result.p[0] if alpha is None else alpha)
    phip = float(amplitude if alpha is None else result.p[0])
    boundary_vector = boundary(result.y[:, 0], result.y[:, -1], result.p)
    if not np.all(np.isfinite(boundary_vector)):
        raise BVPError('non-finite boundary matching residual')
    boundary_residual = float(np.max(np.abs(boundary_vector)))
    if (np.max(result.rms_residuals) > tolerance*1.01 or boundary_residual > 1e-10
            or min(bp, result.p[1], np.min(result.y[0])) <= 0 or coupling <= 0):
        raise BVPError('collocation result failed residual or positivity validation')
    return BranchSolution(result, bp, phip, coupling, *map(float, result.p[1:]),
                          cutoff, order, tolerance, initial_nodes, degree,
                          alpha is not None, boundary_residual)


def exterior_check(solution, rmax=1e5, launch=1e-5, rtol=1e-11):
    """Independent outward integration, used only to check AF compatibility.

    It does not find or seed the two-horizon solution. Positivity and coverage
    are mandatory; the finite-radius diagnostics are not an existence proof
    of the asymptotic limit.
    """
    if (not np.all(np.isfinite([rmax, launch, rtol])) or rmax < 100
            or not 0 < launch < DELTA/4 or not 0 < rtol < 1):
        raise ValueError('invalid exterior domain or precision')
    def rhs(r, y):
        b, bd, phi, pd = y
        p = (r-RP)*(r-RM)
        return [bd, -b*pd**2, pd,
                (solution.alpha*np.sinh(solution.alpha*phi)/b**2-(2*r-RP-RM)*pd)/p]

    def hit_zero(r, y):
        return y[0]-1e-8
    hit_zero.terminal = True
    hit_zero.direction = -1
    start = RP+launch
    initial = taylor_fields(solution.bp, solution.bdp, solution.phip, solution.alpha,
                             DELTA, launch, solution.order)
    try:
        result = solve_ivp(rhs, (start, rmax), initial, method='DOP853',
                           rtol=rtol, atol=rtol/100, events=hit_zero, dense_output=True)
    except (ValueError, FloatingPointError, OverflowError) as exc:
        raise BVPError('exterior raised: '+str(exc)) from exc
    if (not result.success or result.status != 0 or result.t[-1] < rmax
            or not np.all(np.isfinite(result.y)) or np.any(result.y[0] <= 0)):
        raise BVPError('exterior did not complete to requested radius')
    if (np.ndim(result.t) != 1 or result.t.size < 2
            or not np.all(np.isfinite(result.t)) or not np.all(np.diff(result.t) > 0)
            or result.t[0] > start or result.t[-1] < rmax
            or np.shape(result.y) != (4, result.t.size) or not callable(result.sol)):
        raise BVPError('exterior returned an invalid mesh or incomplete interval')
    radii = np.array([rmax/10, rmax])
    values = result.sol(radii)
    b, bd, phi, pd = values
    p = (radii-RP)*(radii-RM)
    h = -1+(2*radii-RP-RM)*bd/b-p*bd**2/b**2+2*np.cosh(solution.alpha*phi)/b**2-p*pd**2
    output = dict(success=True, rmax=float(rmax), lambda_values=bd.tolist(),
                  phi_values=phi.tolist(), b_over_r_values=(b/radii).tolist(),
                  scaled_phi_derivative_values=(radii*pd).tolist(),
                  hamiltonian_max_abs=float(np.max(np.abs(h))),
                  positive_asymptotic_slope=bool(np.all(bd > 0)))
    if not all(np.all(np.isfinite(v)) for v in (values, h)):
        raise BVPError('non-finite exterior diagnostics')
    return output
