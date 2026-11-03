#!/usr/bin/env python3
"""An explicit regular two-horizon black hole with an entropy EXCESS, S+ S- > S_ext^2.

Theory (paper's action, V_g = 0, one canonical scalar, electric-magnetic frame of Sec. 5):
    f_1 = e^{-a1 phi} (electric charge Q),  f_2 = e^{a2 phi} (magnetic charge P),
    V_eff = Q^2 e^{a1 phi} + P^2 e^{a2 phi},   a1 = sqrt(7), a2 = -2 sqrt(7),   Q^2 = 2, P^2 = 1.
Attractor phi_0 = 0, V_eff^0 = 3, S_ext = 3 pi, linear index nu(nu+1) = -a1 a2 / 2 = 7.

Two constructions and a consistency check:
  A. reversible autonomous shooting (time-reversal matching phi'(tau_c) = 0, valid for any V);
  B. radial two-sided shooting with horizon series (cf. App. B), NO reflection assumed;
  C. the coupling identity integrated along the branch from the RN bifurcation at nu = 2 (eta = 1):
     d log eta / ds = (2/(s d)) Int dr phi V'(phi) / b^2   for the family a1 = s, a2 = -2 s at fixed charges.
Then: asymptotically flat exterior for b_+ = 2.5 b_+^DE (r_+ = 1, r_- = 0.35), its mass, scalar charge,
temperatures, potentials and the Smarr relation; Kretschmann scalar at both horizons.
"""
import numpy as np
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq, root

# ----------------------------------------------------------------------------- theory
class Theory:
    def __init__(self, a1, a2, Q2, P2):
        self.a1, self.a2, self.Q2, self.P2 = a1, a2, Q2, P2
        self.phi0 = np.log(-a2*P2/(a1*Q2))/(a1 - a2)
        self.V0 = self.V(self.phi0)

    def V(self, f):
        return self.Q2*np.exp(self.a1*f) + self.P2*np.exp(self.a2*f)

    def dV(self, f):
        return self.a1*self.Q2*np.exp(self.a1*f) + self.a2*self.P2*np.exp(self.a2*f)

    def nu(self):
        lam = (self.a1**2*self.Q2*np.exp(self.a1*self.phi0)
               + self.a2**2*self.P2*np.exp(self.a2*self.phi0))/(2*self.V0)
        return (-1 + np.sqrt(1 + 4*lam))/2


# ----------------------------------------------------------------------------- A: autonomous, d = 1
def half_interval(th, eps, t0=-40.0, rtol=1e-12, dense=False):
    """U'' = -e^{2U} V, phi'' = -e^{2U} V'/2 from the regular outer horizon (U ~ tau/2, b_+ = 1)
    to the unique maximum of U."""
    def rhs(t, y):
        U, Ud, f, fd = y
        e = np.exp(2*U)
        return [Ud, -e*th.V(f), fd, -0.5*e*th.dV(f)]
    E = np.exp(t0)
    y0 = [t0/2 - th.V(eps)*E, 0.5 - th.V(eps)*E, eps - 0.5*th.dV(eps)*E, -0.5*th.dV(eps)*E]
    ev = lambda t, y: y[1]
    ev.terminal, ev.direction = True, -1
    sol = solve_ivp(rhs, (t0, 600.0), y0, method='DOP853', rtol=rtol, atol=rtol*1e-2,
                     events=ev, dense_output=dense)
    if not sol.success or len(sol.t_events[0]) != 1 or not np.all(np.isfinite(sol.y)):
        raise RuntimeError("Regular center was not reached: " + sol.message)
    return sol


def even_mismatch(th, eps, **kw):
    return half_interval(th, eps, **kw).y_events[0][0][3]


def eta_autonomous(th, eps, **kw):
    s = half_interval(th, eps, **kw)
    return (np.exp(-s.t_events[0][0])/th.V0)**2          # b_+ b_- = e^{-tau_c} for d = 1


# ----------------------------------------------------------------------------- B: radial two-sided shooting
def smul(a, b, N):
    c = np.zeros(N)
    for i in range(min(len(a), N)):
        m = min(len(b), N - i)
        c[i:i+m] += a[i]*b[:m]
    return c


def sexp(a, N):
    e = np.zeros(N); e[0] = np.exp(a[0])
    da = np.array([k*a[k] for k in range(len(a))])
    for n in range(1, N):
        e[n] = sum(da[k]*e[n-k] for k in range(1, min(n, len(a)-1) + 1))/n
    return e


def sinv(a, N):
    c = np.zeros(N); c[0] = 1/a[0]
    for n in range(1, N):
        c[n] = -sum(a[k]*c[n-k] for k in range(1, min(n, len(a)-1) + 1))/a[0]
    return c


def horizon_series(th, dh, bh, ph, N):
    """Regular series at a horizon, s = r - r_h, R = dh s + s^2 (recursion of App. B)."""
    b = np.zeros(N + 2); q = np.zeros(N + 2)
    b[0], q[0] = bh, ph
    b[1] = (bh**2 - th.V(ph))/(dh*bh)                    # horizon constraint
    for n in range(N):
        M = n + 2
        Vp = th.a1*th.Q2*sexp(th.a1*q[:M], M) + th.a2*th.P2*sexp(th.a2*q[:M], M)
        src = smul(Vp, sinv(smul(b[:M], b[:M], M), M), M)/2.0
        q[n+1] = (src[n] - n*(n + 1)*q[n])/(dh*(n + 1)**2)
        dq = np.array([(k + 1)*q[k+1] for k in range(M)])
        b[n+2] = -smul(b[:M], smul(dq, dq, M), M)[n]/((n + 2)*(n + 1))
    return b[:N+1], q[:N+1]


def series_value(c, s):
    P = np.polynomial.polynomial
    return P.polyval(s, c), P.polyval(s, P.polyder(c))


def radial_rhs(th, rp, rm, with_potentials=False):
    def f(r, y):
        b, bp, ph, pp = y[:4]
        R = (r - rp)*(r - rm); Rp = 2*r - rp - rm
        out = [bp, -b*pp**2, pp, (th.dV(ph)/(2*b*b) - Rp*pp)/R]
        if with_potentials:
            out += [th.Q2*np.exp(th.a1*ph)/b**2, th.P2*np.exp(th.a2*ph)/b**2]
        return out
    return f


def two_sided(th, B, guess, rp=1.0, rm=0.35, N=8, d_out=1e-3, d_in=1e-4, rtol=1e-13, dense=False):
    d = rp - rm; xm = 0.5*(rp + rm); f = radial_rhs(th, rp, rm)

    def sides(x, dense=False):
        eps, bm, pm = x
        bo, qo = horizon_series(th, +d, B, eps, N)
        bi, qi = horizon_series(th, -d, bm, pm, N)
        yb, yq = series_value(bo, -d_out), series_value(qo, -d_out)
        so = solve_ivp(f, (rp - d_out, xm), [yb[0], yb[1], yq[0], yq[1]], method='DOP853',
                       rtol=rtol, atol=1e-16, dense_output=dense)
        zb, zq = series_value(bi, d_in), series_value(qi, d_in)
        si = solve_ivp(f, (rm + d_in, xm), [zb[0], zb[1], zq[0], zq[1]], method='DOP853',
                       rtol=rtol, atol=1e-16, dense_output=dense)
        if not (so.success and si.success) or abs(so.t[-1]-xm)>1e-12 or abs(si.t[-1]-xm)>1e-12:
            raise RuntimeError("Incomplete two-sided integration")
        if not all(np.all(np.isfinite(q.y)) and np.min(q.y[0])>0 for q in (so,si)):
            raise RuntimeError("Invalid two-sided fields")
        return so, si

    def F(x):
        so, si = sides(x)
        yo, yi = so.y[:, -1], si.y[:, -1]
        return [yo[0] - yi[0], yo[2] - yi[2], yo[3] - yi[3]]   # b' mismatch left as a check

    sol = root(F, guess, method='hybr', options=dict(xtol=1e-15, eps=1e-9))
    so, si = sides(sol.x, dense=dense)
    if np.max(np.abs(sol.fun)) > 5e-9:
        raise RuntimeError("Two-sided root did not converge")
    return sol.x, so, si, np.max(np.abs(sol.fun)), so.y[1, -1] - si.y[1, -1]


def exterior(th, B, eps, rp=1.0, rm=0.35, N=10, delta=1e-3, rmax=4e6, rtol=1e-13):
    d = rp - rm
    bo, qo = horizon_series(th, +d, B, eps, N)
    yb, yq = series_value(bo, delta), series_value(qo, delta)
    g1 = lambda ph, b: th.Q2*np.exp(th.a1*ph)/b**2
    g2 = lambda ph, b: th.P2*np.exp(th.a2*ph)/b**2
    I10 = 0.5*delta*(g1(eps, B) + g1(yq[0], yb[0]))
    I20 = 0.5*delta*(g2(eps, B) + g2(yq[0], yb[0]))
    sol = solve_ivp(radial_rhs(th, rp, rm, True), (rp + delta, rmax),
                    [yb[0], yb[1], yq[0], yq[1], I10, I20], method='DOP853', rtol=rtol, atol=1e-30,
                    dense_output=True)
    if not sol.success or sol.t[-1] != rmax or not np.all(np.isfinite(sol.y)) or np.min(sol.y[0]) <= 0:
        raise RuntimeError("Incomplete or invalid exterior")
    # Richardson in 1/r at r = rmax/4, rmax/2:  b - r b' = beta + 2 gamma / r + ...
    rs = [rmax/4, rmax/2]
    vals = [sol.sol(r) for r in rs]
    beta = 2*(vals[1][0] - rs[1]*vals[1][1]) - (vals[0][0] - rs[0]*vals[0][1])
    c = vals[1][1]
    phinf = vals[1][2] + rs[1]*vals[1][3]
    phi1 = 2*(-rs[1]**2*vals[1][3]) - (-rs[0]**2*vals[0][3])
    tail = lambda v, r: (v[4] + g1(v[2], v[0])*r, v[5] + g2(v[2], v[0])*r)
    t0_, t1_ = tail(vals[0], rs[0]), tail(vals[1], rs[1])
    I1, I2 = 2*t1_[0] - t0_[0], 2*t1_[1] - t0_[1]
    return dict(c=c, M=c*(rp + rm)/2 + beta, phinf=phinf, Sigma=c*phi1,
                T=c*d/(4*np.pi*B**2), QPhi=c*I1, PPsi=c*I2, sol=sol, min_bprime=np.min(sol.y[1]))


def kretschmann_at_horizon(th, dh, bh, ph):
    b1 = (bh**2 - th.V(ph))/(dh*bh)
    fp = dh/bh**2
    fpp = 2/bh**2 - 4*dh*b1/bh**3
    return fpp**2 + 4*(fp*b1/bh)**2 + 4/bh**4


# ----------------------------------------------------------------------------- C: coupling identity along the branch
def coupling_identity_eta(s_end, n_nodes=20):
    """eta at a1 = s_end from Int_{sqrt6}^{s_end} ds (2/s) Int e^{2U} phi V' dtau (d = 1), RN bifurcation at nu = 2."""
    s0 = np.sqrt(6.0)
    xg, wg = np.polynomial.legendre.leggauss(n_nodes)
    ss = 0.5*(s_end - s0)*xg + 0.5*(s_end + s0); ws = 0.5*(s_end - s0)*wg
    log_eta = 0.0
    for s, w in zip(ss, ws):
        th = Theory(s, -2*s, 2.0, 1.0)
        nu = th.nu()
        guess = max((nu - 2)/0.42, 1e-4)        # nu - 2 = (s^3/35) eps + ... near the bifurcation
        eps = brentq(lambda e: even_mismatch(th, e), 0.5*guess, 1.6*guess, xtol=1e-15, rtol=1e-15)
        sol = half_interval(th, eps, dense=True)
        tc = sol.t_events[0][0]
        g = lambda t: (lambda y: np.exp(2*y[0])*y[2]*th.dV(y[2]))(sol.sol(t))
        J = quad(g, sol.t[0], tc, limit=400, epsabs=1e-14, epsrel=1e-13)[0]
        log_eta += w*(2.0/s)*2*J                 # even branch: full interval = 2 x half interval
    return np.exp(log_eta)


if __name__ == '__main__':
    th = Theory(np.sqrt(7), -2*np.sqrt(7), 2.0, 1.0)
    Sext = np.pi*th.V0
    print(f"Theory: a1 = sqrt7, a2 = -2 sqrt7, Q^2 = 2, P^2 = 1;  phi0 = {th.phi0:.1f}, V0 = {th.V0:.1f}, "
          f"S_ext = 3 pi, nu = {th.nu():.12f} = (sqrt29 - 1)/2")

    # A
    print("\nA. reversible autonomous shooting (even matching):")
    for t0, rt in [(-40.0, 1e-12), (-50.0, 1e-13)]:
        eps = brentq(lambda e: even_mismatch(th, e, t0=t0, rtol=rt), 0.40, 0.45, xtol=1e-15, rtol=1e-15)
        print(f"   tau0 = {t0:5.1f}, rtol = {rt:g}:  phi_+ = {eps:.13f}   eta = {eta_autonomous(th, eps, t0=t0, rtol=rt):.13f}")
    epsA = eps; etaA = eta_autonomous(th, eps, t0=-50.0, rtol=1e-13)

    # B
    rp, rm = 1.0, 0.35; d = rp - rm
    bDE = (d + np.sqrt(d*d + 4*th.V0))/2          # RN outer radius at the same coordinate separation
    B = 2.5*bDE
    print(f"\nB. radial two-sided shooting, r_+ = 1, r_- = 0.35, b_+ = 2.5 b_+^DE = {B:.12f}:")
    guess = [epsA, np.sqrt(etaA)*th.V0/B, epsA]
    for N, do, di in [(6, 1e-3, 1e-4), (8, 1e-3, 1e-4), (10, 1e-3, 3e-5)]:
        x, so, si, res, dbp = two_sided(th, B, guess, N=N, d_out=do, d_in=di)
        print(f"   N = {N:2d}: phi_+ = {x[0]:.13f}  phi_- = {x[2]:.13f}  b_- = {x[1]:.13f}  "
              f"eta = {(B*x[1]/th.V0)**2:.13f}  |F| = {res:.0e}  unused b' mismatch = {dbp:+.0e}")
    epsB, bm, pm = x
    etaB = (B*bm/th.V0)**2

    # C
    print("\nC. coupling identity integrated from the RN bifurcation (nu = 2, eta = 1):")
    print(f"   eta = {coupling_identity_eta(np.sqrt(7)):.13f}")

    # exterior and thermodynamics
    ext = exterior(th, B, epsB)
    Sp, Sm = np.pi*B**2, np.pi*bm**2
    Tm = -ext['c']*d/(4*np.pi*bm**2)
    smarr = ext['M'] - (2*ext['T']*Sp + ext['QPhi'] + ext['PPsi'])
    k = th.V(epsB)/th.V0
    print("\nAsymptotically flat representative (G = 1, unit lapse at infinity):")
    print(f"   M = {ext['M']:.9f}   phi_inf = {ext['phinf']:.9f}   Sigma = {ext['Sigma']:.9f}   (min b' = {ext['min_bprime']:.4f} > 0)")
    print(f"   S_+ = {Sp:.9f}   S_- = {Sm:.9f}   S_+ S_- = {Sp*Sm:.9f}   S_ext^2 = 9 pi^2 = {Sext**2:.9f}")
    print(f"   eta = S_+ S_- / S_ext^2 = {Sp*Sm/Sext**2:.12f}")
    print(f"   T_+ = {ext['T']:.10f}   T_- = {Tm:.10f}   T_+ S_+ = {ext['T']*Sp:.9f}   T_- S_- = {Tm*Sm:.9f}")
    print(f"   Q Phi_+ = {ext['QPhi']:.9f}   P Psi_+ = {ext['PPsi']:.9f}   Smarr residual M - (2 T_+ S_+ + Q Phi_+ + P Psi_+) = {smarr:.1e}")
    print(f"   k = V(phi_+)/V0 = {k:.6f};  orbit floor (k - eta/k) = {k - etaB/k:.6f} < (S_+ - S_-)/S_ext = {(Sp - Sm)/Sext:.6f}")
    print(f"   Kretschmann: outer horizon {kretschmann_at_horizon(th, d, B, epsB):.6e},  inner horizon {kretschmann_at_horizon(th, -d, bm, pm):.6e}")

    # Explicit acceptance gates for the fixed-theory example.
    assert abs(etaA-etaB)<3e-11 and etaB>1.02
    assert abs(epsB-pm)<1e-9 and abs(dbp)<1e-8
    assert abs(smarr)<2e-7 and ext['min_bprime']>0
    print('[PASS] Regular excess example, two constructions and asymptotically flat exterior')
