#!/usr/bin/env python3
"""Entropy products on regular two-horizon branches with UNEQUAL exponential slopes.

Model (unit normalisation of the paper, attractor at phi = 0, V_eff^0 = 2, S_ext = 2*pi):
    V(phi) = 2/(1+r) * [ r*exp(s*phi) + exp(-r*s*phi) ]
i.e. slopes (alpha_1, alpha_2) = (s, -r*s) with charges Q_1^2 = 2r/(1+r), Q_2^2 = 2/(1+r).
r = 1 is the symmetric model 2*cosh(alpha*phi); r = 3/2 at nu = 3 is the B2 Toda pair (4,-6).
Linear index: nu(nu+1) = V''(0)/4 = r*s^2/2.

Reversible variables of Sec. 4.1 with d = 1 (outer horizon at tau -> -inf, U = tau/2 + 0):
    U'' = -e^{2U} V,   phi'' = -(1/2) e^{2U} V',   U'^2 + phi'^2 + e^{2U} V = 1/4.
Matching conditions (branch traces solve for s at prescribed eps = phi_+):
    'even' : phi'(tau_c) = 0 at U'(tau_c) = 0  (time reversal only -- valid for ANY V)
    'odd'  : phi(tau_c)  = 0 at U'(tau_c) = 0  (needs V even: r = 1 only)
    'full' : phi'(+inf) = 0, i.e. regular inner horizon, no reflection assumed
eta = S_+ S_- / S_ext^2 = (b_+ b_- / V0)^2 = (e^{-u1}/2)^2, u1 = lim (U + tau/2) at the inner end
      (= tau_c for the reflected branches).
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq


def potential(s, r):
    A, B = 2*r/(1+r), 2/(1+r)
    V = lambda f: A*np.exp(s*f) + B*np.exp(-r*s*f)
    dV = lambda f: A*s*np.exp(s*f) - B*r*s*np.exp(-r*s*f)
    return V, dV


def nu_of(s, r):
    return (-1 + np.sqrt(1 + 2*r*s*s))/2


def s_of(nu, r):
    return np.sqrt(2*nu*(nu+1)/r)


def integrate(s, r, eps, to_turning_point, t0=-40.0, rtol=1e-12):
    V, dV = potential(s, r)

    def rhs(t, y):
        U, Ud, f, fd = y
        e = np.exp(2*U)
        return [Ud, -e*V(f), fd, -0.5*e*dV(f)]

    E = np.exp(t0)  # outer-horizon series to first order in e^{tau}
    y0 = [t0/2 - V(eps)*E, 0.5 - V(eps)*E, eps - 0.5*dV(eps)*E, -0.5*dV(eps)*E]
    if to_turning_point:
        ev = lambda t, y: y[1]
    else:
        ev = lambda t, y: np.exp(2*y[0])*V(y[2]) - 1e-26
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(rhs, (t0, 600.0), y0, method='DOP853', rtol=rtol, atol=rtol*1e-2, events=ev)
    if not sol.success or len(sol.t_events[0]) != 1 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError("Incomplete asymmetric integration: " + sol.message)
    t, y = sol.t[-1], sol.y[:, -1]
    H = y[1]**2 + y[3]**2 + np.exp(2*y[0])*V(y[2]) - 0.25
    return t, y, H


def mismatch(s, r, eps, kind):
    if kind == "odd" and r != 1:
        raise ValueError("Odd scalar reflection requires an even potential")
    if kind == 'full':
        t, y, _ = integrate(s, r, eps, False)
        return y[3]
    t, y, _ = integrate(s, r, eps, True)
    return y[3] if kind == 'even' else y[2]


def solve(r, eps, kind, s_guess):
    g = lambda s: mismatch(s, r, eps, kind)
    for w in (0.002, 0.005, 0.01, 0.02, 0.05, 0.1):
        grid = s_guess*np.linspace(1 - w, 1 + w, 21)
        vals = [g(x) for x in grid]
        brackets = [(grid[i], grid[i+1]) for i in range(20) if np.sign(vals[i]) != np.sign(vals[i+1])]
        if brackets:
            lo, hi = min(brackets, key=lambda b: abs(0.5*(b[0] + b[1]) - s_guess))
            return brentq(g, lo, hi, xtol=1e-14, rtol=1e-14)
    raise RuntimeError(f"no root near s={s_guess} (r={r}, eps={eps})")


def products(s, r, eps, kind):
    if kind == 'full':
        t, y, H = integrate(s, r, eps, False)
        u1, phim = y[0] + t/2, y[2]
    else:
        t, y, H = integrate(s, r, eps, True)
        u1 = t
        phim = eps if kind == 'even' else -eps
    return (np.exp(-u1)/2)**2, phim, H


def branch(r, n, kind, eps_list, label):
    print(f"\n{label}:  r = {r}, mode n = {n}, matching = '{kind}'")
    print(f"{'eps':>7} {'nu':>12} {'nu-n':>11} {'phi_-':>10} {'100(eta-1)':>12} {'|H|':>8}")
    s_prev = s_of(n, r)
    for eps in eps_list:
        s = solve(r, eps, kind, s_prev)
        s_prev = s
        eta, phim, H = products(s, r, eps, kind)
        nu = nu_of(s, r)
        print(f"{eps:+7.3f} {nu:12.8f} {nu-n:+11.3e} {phim:+10.6f} {100*(eta-1):+12.6f} {abs(H):8.1e}")


if __name__ == '__main__':
    # 1. Paper's Table 1 (symmetric, three nodes): expect 100(1-eta) = 0.3573, 1.9178, 3.4901
    branch(1.0, 3, 'odd', [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0], "Symmetric model, check of Table 1")
    # 2. B2 Toda control at nu = 3 (alpha_1, alpha_2) = (4, -6): expect nu = 3, eta = 1 exactly
    branch(1.5, 3, 'full', [0.05, 0.2, 0.5], "B2 Toda control")
    # 3. Three nodes, slopes between symmetric and B2: deficit
    branch(1.25, 3, 'full', [0.05, 0.2, 0.5], "Unequal slopes below the B2 ratio")
    # 4. Three nodes, slopes beyond B2: branch opens to LARGER nu, eta > 1 for both signs of eps
    branch(2.0, 3, 'full', [0.1, 0.3, 0.5, 1.0], "Unequal slopes beyond the B2 ratio (eps > 0)")
    branch(2.0, 3, 'full', [-0.1, -0.3, -0.5, -1.0], "Unequal slopes beyond the B2 ratio (eps < 0)")
    # 5. Two nodes (even, time-reversal reflection): nu - 2 linear in eps, excess for eps > 0.
    #    Leading slope: nu - n = -(V'''(0)/(4 V0)) (n n n; 0 0 0)^2 eps; for r = 2, n = 2 this is 0.4199 eps,
    #    and the coupling identity gives eta - 1 = (2/3)(dnu/deps) eps^3 = 0.280 eps^3.
    branch(2.0, 2, 'even', [0.01, 0.1, 0.5, 1.0, 2.0, 3.0], "Two nodes, unequal slopes (eps > 0)")
    branch(2.0, 2, 'even', [-0.01, -0.1, -0.5, -1.0, -2.0, -3.0], "Two nodes, unequal slopes (eps < 0)")
