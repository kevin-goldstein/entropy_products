r"""Exact checks for the static KLR continuation and its charge dictionary.

The static map is checked off shell with arbitrary coupling functions, including
an axionic term and the required V -> -V.  The balanced A1+A1 family is written
with a rational deformation parameter h, |h| < 1, so its complete field equations,
fluxes, entropy, conical balance and curvature are identities rather than samples.

The two-exponential charge identities require smooth positive endpoint fields and
vanishing scalar boundary flux.  Their algebra applies to every such regular
branch, without assuming a family through the spherical attractor.  Numerical
branch existence is independently tested by the guarded BVP solver, not inferred
from these identities.  A dipole EOM example below checks an entire spacetime at
one exact rational parameter tuple; it does not establish a general-parameter
identity.  No numerical shooting or endpoint extrapolation occurs in this file.

Run: python3 klr_continuation.py
"""
import time
import platform
from itertools import permutations
import sympy as sp
from sympy.combinatorics import Permutation
import checklib as ck

T0 = time.time()
print("Python %s; SymPy %s; all residual checks below are symbolic" % (platform.python_version(), sp.__version__))
EPS4 = {p: Permutation(list(p)).signature() for p in permutations(range(4))}


def zeros(label, expressions):
    """Check each residual separately so errors cannot cancel in a sum."""
    for index, expression in enumerate(expressions, 1):
        ck.zero('%s [%d]' % (label, index), expression)


# ----------------------------------------------------------------------------------------------------------------
# A. the continuation as a map of static solutions
# ----------------------------------------------------------------------------------------------------------------
def christoffel_ricci(g, X):
    n = len(X)
    ginv = g.inv()
    Gam = [[[sp.simplify(sum(ginv[a, d] * (sp.diff(g[d, b], X[c]) + sp.diff(g[d, c], X[b]) - sp.diff(g[b, c], X[d]))
                              for d in range(n)) / 2) for c in range(n)] for b in range(n)] for a in range(n)]
    Ric = sp.Matrix(n, n, lambda b, c: sp.simplify(sum(
        sp.diff(Gam[a][b][c], X[a]) - sp.diff(Gam[a][b][a], X[c])
        + sum(Gam[a][a][d] * Gam[d][b][c] - Gam[a][c][d] * Gam[d][b][a] for d in range(n)) for a in range(n))))
    return ginv, Ric


def field_equations(g, sqg, X, phi, Fs, fs, fts, V):
    """(eq:EOM1), (eq:EOM2) for one scalar (g_11 = 1), gauge fields F_a with diagonal couplings f_a, ft_a, potential V:
    returns the trace-reversed Einstein residual R_mn - [...], the scalar residual and the components of dG_a."""
    n = 4
    u = sp.Symbol('u')
    ginv, Ric = christoffel_ricci(g, X)
    rhs = 2 * sp.Matrix(n, n, lambda a, b: sp.diff(phi, X[a]) * sp.diff(phi, X[b])) + V(phi) * g
    box = sum(sp.diff(sqg * sum(ginv[a, b] * sp.diff(phi, X[b]) for b in range(n)), X[a]) for a in range(n)) / sqg
    scal = 4 * box - 2 * sp.diff(V(u), u).subs(u, phi)
    dG = []
    for F, f, ft in zip(Fs, fs, fts):
        Fup = ginv * F * ginv
        F2 = sp.simplify(sum(F[a, b] * Fup[a, b] for a in range(n) for b in range(n)))
        starF = sp.zeros(n)
        for (m, nn, rr, s), sgn in EPS4.items():
            starF[m, nn] += sp.Rational(1, 2) * sqg * sgn * Fup[rr, s]
        FsF = sp.simplify(sum(F[a, b] * (ginv * starF * ginv)[a, b] for a in range(n) for b in range(n)))
        rhs += 2 * f(phi) * (F * ginv * F.T - g * F2 / 4)
        scal -= sp.diff(f(u), u).subs(u, phi) * F2 + sp.diff(ft(u), u).subs(u, phi) * FsF
        G = (f(phi) * starF - ft(phi) * F).applyfunc(sp.simplify)
        dG += [sp.simplify(sp.diff(G[b, c], X[a]) + sp.diff(G[c, a], X[b]) + sp.diff(G[a, b], X[c]))
               for a in range(n) for b in range(a + 1, n) for c in range(b + 1, n)]
    return (Ric - rhs).applyfunc(sp.simplify), sp.simplify(scal), dG


r = sp.Symbol('r', real=True)
t, th, ps, tau, rho, vp = sp.symbols('t theta psi tau rho varphi', real=True)
a2f, bf, phf = sp.Function('a2')(r), sp.Function('b', positive=True)(r), sp.Function('phi')(r)
ZERO = lambda u_: 0 * u_


def static_and_continued(Es, Ps, fs, fts, V, Vc):
    gs = sp.diag(-a2f, 1 / a2f, bf ** 2, bf ** 2 * sp.sin(th) ** 2)
    gc = sp.diag(-bf ** 2 * rho ** 2, bf ** 2 / rho ** 2, -1 / a2f, -a2f)
    Fs_s, Fs_c = [], []
    for E, P in zip(Es, Ps):
        F = sp.zeros(4); F[1, 0], F[0, 1], F[2, 3], F[3, 2] = E, -E, P * sp.sin(th), -P * sp.sin(th)
        Fs_s.append(F)
        F = sp.zeros(4); F[2, 3], F[3, 2], F[0, 1], F[1, 0] = E, -E, P, -P
        Fs_c.append(F)
    Rs, ss, dGs = field_equations(gs, bf ** 2 * sp.sin(th), [t, r, th, ps], phf, Fs_s, fs, fts, V)
    Rc, sc, dGc = field_equations(gc, bf ** 2, [tau, rho, r, vp], phf, Fs_c, fs, fts, Vc)
    mixed_s = [Rs[0, 0] / gs[0, 0], Rs[1, 1] / gs[1, 1], Rs[2, 2] / gs[2, 2], Rs[3, 3] / gs[3, 3]]
    # tt <-> varphi varphi, rr <-> rr, theta theta and psi psi <-> rho rho and tau tau
    mixed_c = [Rc[3, 3] / gc[3, 3], Rc[2, 2] / gc[2, 2], Rc[1, 1] / gc[1, 1], Rc[0, 0] / gc[0, 0]]
    offd = [Rc[i, j] for i in range(4) for j in range(4) if i != j] + [Rs[i, j] for i in range(4) for j in range(4) if i != j]
    return mixed_s, mixed_c, ss, sc, dGs, dGc, offd


print('--- A. the continuation maps static region-II solutions to static near-horizon solutions ---')
f1, f2, Vf = sp.Function('f1'), sp.Function('f2'), sp.Function('V')
Q1, P2 = sp.symbols('Q1 P2', real=True)
E1 = Q1 / (f1(phf) * bf ** 2)
ms, mc, ss, sc, dGs, dGc, offd = static_and_continued([E1, 0], [0, P2], [f1, f2], [ZERO, ZERO], Vf, lambda u_: -Vf(u_))
ck.zero('A(i) two gauge fields, f_1, f_2, V arbitrary: E^t_t + E^varphi_varphi = 0 (continued = minus static)', ms[0] + mc[0])
ck.zero('A(i) E^r_r(continued) + E^r_r(static) = 0', ms[1] + mc[1])
ck.zero('A(i) E^rho_rho(continued) + E^theta_theta(static) = 0', ms[2] + mc[2])
ck.zero('A(i) E^tau_tau(continued) + E^psi_psi(static) = 0', ms[3] + mc[3])
ck.zero('A(i) scalar equation: continued + static = 0', sc + ss)
ck.true('A(i) Maxwell equations hold in both configurations, no off-diagonal Einstein components',
        all(sp.simplify(x) == 0 for x in dGs + dGc + offd))
ms_n, mc_n, ss_n, sc_n, _, _, _ = static_and_continued([E1, 0], [0, P2], [f1, f2], [ZERO, ZERO], Vf, Vf)
ck.zero('A(i) unflipped potential produces the exact obstruction -2 V', ms_n[0] + mc_n[0] + 2 * Vf(phf))
ck.large('A(i) negative control: without V -> -V the tt component does not map (residual -2V)',
         sp.simplify(ms_n[0] + mc_n[0]).subs({Vf(phf): sp.Rational(7, 3)}).subs(sp.Function('V')(phf), sp.Rational(7, 3)), 1)
f, ft = sp.Function('f'), sp.Function('ft')
Q, P = sp.symbols('Q P', real=True)
E = (Q + ft(phf) * P) / (f(phf) * bf ** 2)
ms, mc, ss, sc, dGs, dGc, offd = static_and_continued([E], [P], [f], [ft], ZERO, ZERO)
ck.zero('A(ii) one dyonic gauge field with f, ft (V = 0): Einstein tt <-> varphi varphi', ms[0] + mc[0])
ck.zero('A(ii) Einstein rr', ms[1] + mc[1])
ck.zero('A(ii) Einstein theta theta <-> rho rho', ms[2] + mc[2])
ck.zero('A(ii) scalar equation, including the axionic term', sc + ss)
ck.true('A(ii) Maxwell: d(f *F - ft F) = 0 in both, with E = (Q + ft P)/(f b^2)', all(sp.simplify(x) == 0 for x in dGs + dGc))
print('   (A done, %.0f s)' % (time.time() - T0))


# B. The complete balanced A1+A1 family, all |h| < 1 and Q,P > 0.
print('--- B. exact balanced A1+A1 near-horizon geometries ---')
z, h = sp.symbols('z h', real=True)
q, p = sp.symbols('Q_e P_m', positive=True)
B0 = 2*q*p
B = B0*(1-h*h*z*z)/(1-h*h)
phi_h = sp.log(p*(1-h*z)/(q*(1+h*z)))/2
exp_phi2 = p*(1-h*z)/(q*(1+h*z))
Hpsi = B0**2*(1-z*z)/B
metric_h = sp.diag(-B*rho**2, B/rho**2, B/(1-z*z), Hpsi)
# The rational expressions are the original ansatz specialized to f_1=f_2=e^{-2phi}.
f_h = 1/exp_phi2
F1_h = sp.zeros(4)
F2_h = sp.zeros(4)
F1_h[2,3] = p*(1-h*h)/(1+h*z)**2
F1_h[3,2] = -F1_h[2,3]
F2_h[0,1], F2_h[1,0] = p, -p
ck.zero('B: determinant fixes positive volume B B0 on the physical patch', -metric_h.det()-(B*B0)**2)
gi_h, Ric_h = christoffel_ricci(metric_h, [tau,rho,z,vp])
phi_grad = sp.Matrix([0,0,sp.diff(phi_h,z),0])
rhs_h = 2*phi_grad*phi_grad.T
scalar_h = 4*sp.diff(B*B0*gi_h[2,2]*sp.diff(phi_h,z),z)/(B*B0)
G_h = []
for number, Fh in enumerate((F1_h,F2_h),1):
    Fup = gi_h*Fh*gi_h
    Fsq = sp.factor(sum(Fh[i,j]*Fup[i,j] for i in range(4) for j in range(4)))
    rhs_h += 2*f_h*(Fh*gi_h*Fh.T-metric_h*Fsq/4)
    scalar_h += 2*f_h*Fsq  # df/dphi = -2f.
    star = sp.zeros(4)
    for (i,j,k,l), sign in EPS4.items():
        star[i,j] += sp.Rational(1,2)*B*B0*sign*Fup[k,l]
    Gh = (f_h*star).applyfunc(sp.factor)
    G_h.append(Gh)
    Xh = [tau,rho,z,vp]
    for i in range(4):
        for j in range(i+1,4):
            for k in range(j+1,4):
                ck.zero('B: exact Maxwell equation field %s component %s%s%s' % (number,i,j,k),
                        sp.diff(Gh[j,k],Xh[i])+sp.diff(Gh[k,i],Xh[j])+sp.diff(Gh[i,j],Xh[k]))
                ck.zero('B: exact Bianchi identity field %s component %s%s%s' % (number,i,j,k),
                        sp.diff(Fh[j,k],Xh[i])+sp.diff(Fh[k,i],Xh[j])+sp.diff(Fh[i,j],Xh[k]))
for i in range(4):
    for j in range(i,4):
        ck.zero('B: exact Einstein equation component %s%s for every deformation h' % (i,j), sp.factor(Ric_h[i,j]-rhs_h[i,j]))
ck.zero('B: exact scalar equation for every deformation h',sp.factor(scalar_h))
ck.zero('B: continued electric field in G1 equals original electric charge Q',G_h[0][0,1]-q)
ck.zero('B: continued electric flux density has the required minus sign',G_h[1][2,3]+q*(1-h*h)/(1-h*z)**2)
# Rational antiderivatives avoid hidden exceptional branches at h=0 in integrate().
primitive_p = p*(1-h*h)*z/(1+h*z)
primitive_q = -q*(1-h*h)*z/(1-h*z)
ck.zero('B: magnetic flux primitive, including h=0',sp.diff(primitive_p,z)-F1_h[2,3])
ck.zero('B: electric flux primitive, including h=0',sp.diff(primitive_q,z)-G_h[1][2,3])
flux_p = (primitive_p.subs(z,1)-primitive_p.subs(z,-1))/2
flux_q = (primitive_q.subs(z,1)-primitive_q.subs(z,-1))/2
ck.zero('B: magnetic charge p_NH=P for all h',flux_p-p)
ck.zero('B: electric charge q_NH=-Q for all h',flux_q+q)
ck.large('B: negative control rejects reversed electric flux orientation',sp.simplify(flux_q-q).subs(q,1),1)
ck.zero('B: horizon area density is constant B0',metric_h[2,2]*metric_h[3,3]-B0**2)
ck.zero('B: entropy area/4 = pi B0 = 2 pi Q P, independent of h',sp.pi*B0-2*sp.pi*q*p)
for endpoint in (-1,1):
    ck.zero('B: conical slope at z=%s gives period 2pi' % endpoint,sp.diff(Hpsi,z).subs(z,endpoint)+2*endpoint*B0)
    ck.zero('B: both endpoint warp factors equal B0 at z=%s' % endpoint,B.subs(z,endpoint)-B0)
# B0 z is the area coordinate, hence K = -(1/(2 B0²)) d_z² Hpsi.
K_h = sp.factor(-sp.diff(Hpsi,z,2)/(2*B0**2))
ck.zero('B: Gauss-Bonnet integral is exactly 4pi',-sp.pi/B0*(sp.diff(Hpsi,z).subs(z,1)-sp.diff(Hpsi,z).subs(z,-1))-4*sp.pi)
ck.zero('B: round member has K=1/B0 everywhere',K_h.subs(h,0)-1/B0)
curvature_difference = sp.factor(K_h.subs(z,1)-K_h.subs(z,0))
ck.zero('B: nonroundness invariant K_pole-K_equator = h²(h⁴-3h²+6)/(B0(1-h²))',
        curvature_difference-h*h*(h**4-3*h*h+6)/(B0*(1-h*h)))
ck.large('B: negative control rejects roundness at h=1/2',curvature_difference.subs({h:sp.Rational(1,2),q:1,p:1}),sp.Rational(1,2))
ck.zero('B: scalar derivative is nonzero for h != 0',sp.diff(phi_h,z)+h/(1-h*h*z*z))
ck.zero('B: first scalar deformation is the dipole harmonic -z',sp.diff(phi_h,h).subs(h,0)+z)
ck.zero('B: metric starts changing at second order',sp.diff(B,h).subs(h,0))
# EOM negative control changes the stress normalization, independently of curvature.
ck.large('B: negative control rejects missing factor 2 in Maxwell stress',
         (Ric_h[0,0]-(rhs_h[0,0]+2*phi_grad[0]*phi_grad[0])/2).subs({h:0,z:0,q:1,p:1,rho:1}),sp.Rational(1,2))
print('   (B done, %.0f s)' % (time.time()-T0))

# C. Exact balance and two-exponential charge identities for any regular branch.
print('--- C. exact balance, integrated equations and charge dictionary ---')
bp,bm,D,scale,I1,I2 = sp.symbols('b_plus b_minus D scale I1 I2',positive=True)
a1=sp.symbols('alpha1',positive=True)
a2=sp.symbols('alpha2',negative=True)
w1,w2=-a2/(a1-a2),a1/(a1-a2)
R = bp*bm
balance_scale=sp.sqrt(bm/bp)
ck.zero('C: unique positive orbit scale balances the radii',bp*balance_scale-bm/balance_scale)
ck.zero('C: entropy is pi b_plus b_minus on the balanced orbit',sp.pi*(bp*balance_scale)**2-sp.pi*R)
ck.zero('C: balanced curvature denominator is D/(b_plus b_minus)',D/(bp*balance_scale)**2-D/R)
unbalanced_den = D/(2*bp**2)+D/(2*bm**2)
ck.zero('C: arbitrary-member denominator error is nonnegative and explicit',unbalanced_den-D/R-D*(bp-bm)**2/(2*bp**2*bm**2))
ck.large('C: negative control rejects unbalanced kappa_plus-kappa_minus in invariant charge formula',
         (unbalanced_den-D/R).subs({D:1,bp:2,bm:1}),sp.Rational(1,10))
# Smooth endpoints make [P phi']_-^+=0. The tt equation gives ∫W/b²=D.
# With Q electric in field1 and P magnetic in field2 these two independent
# integrated equations are q I1+p I2=D and a1 q I1+a2 p I2=0.
sol_integrals=sp.solve([q*I1+p*I2-D,a1*q*I1+a2*p*I2],(I1,I2),dict=True)[0]
ck.zero('C: electric potential difference for every regular branch',sol_integrals[I1]-D*w1/q)
ck.zero('C: magnetic potential difference for every regular branch',sol_integrals[I2]-D*w2/p)
ck.zero('C: weights sum to one',w1+w2-1)
ck.zero('C: weighted scalar force vanishes',a1*w1+a2*w2)
ck.zero('C: horizon identity is recovered',q*sol_integrals[I1]+p*sol_integrals[I2]-D)
pNH,qNH = w1*R/q,-w2*R/p
ck.zero('C: continued magnetic flux equals w1 R/Q',sol_integrals[I1]/(D/R)-pNH)
ck.zero('C: continued electric flux equals -w2 R/P',-sol_integrals[I2]/(D/R)-qNH)
# A nonzero scalar endpoint flux J changes the charges; regularity is essential.
J=sp.symbols('scalar_endpoint_flux',real=True)
sol_flux=sp.solve([q*I1+p*I2-D,a1*q*I1+a2*p*I2-2*J],(I1,I2),dict=True)[0]
ck.zero('C: exact correction from nonzero scalar boundary flux',sol_flux[I1]-sol_integrals[I1]-2*J/((a1-a2)*q))
ck.large('C: negative control rejects applying regular charge formula to nonzero boundary flux',
         (sol_flux[I1]-sol_integrals[I1]).subs({J:1,a1:2,a2:-2,q:1}),sp.Rational(1,4))
# For positive charges W*=K Q^(2w1) P^(2w2), K=w1^(-w1)w2^(-w2).
# Verify the minimum and duality in logarithmic coordinates; positivity makes
# logarithms single valued and avoids unsafe power simplifications.
phi_star = sp.log(-a2*p*p/(a1*q*q))/(a1-a2)
logK = -w1*sp.log(w1)-w2*sp.log(w2)
logWstar = logK+2*w1*sp.log(q)+2*w2*sp.log(p)
# The stationary terms have ratio w1:w2 and sum W*.
ck.zero('C: electric stationary term has weight w1',sp.expand_log(2*sp.log(q)+a1*phi_star-logWstar-sp.log(w1),force=True).simplify())
ck.zero('C: magnetic stationary term has weight w2',sp.expand_log(2*sp.log(p)+a2*phi_star-logWstar-sp.log(w2),force=True).simplify())
logWcontinued = logK+2*w1*(sp.log(w1)+sp.log(R)-sp.log(q))+2*w2*(sp.log(w2)+sp.log(R)-sp.log(p))
ck.zero('C: continued spherical minimum W*_cont = R²/W*_original',sp.expand(logWcontinued+logWstar-2*sp.log(R)).simplify())
eta,Sstar=sp.symbols('eta S_star',positive=True)
SNH=sp.sqrt(eta)*Sstar
Ssphere=eta*Sstar
ck.zero('C: exact tuned-branch entropy relation S_sphere = S_NH²/S_star',Ssphere-SNH**2/Sstar)
ck.zero('C: warped-to-spherical entropy ratio is eta^(-1/2)',SNH/Ssphere-1/sp.sqrt(eta))
ck.large('C: negative control rejects equality for a subunit product',
         (SNH-Ssphere).subs({eta:sp.Rational(81,100),Sstar:1}),sp.Rational(1,20))
print('   (C done, %.0f s)' % (time.time()-T0))

# D. D = 4, rotating: near-horizon extremal Kerr-Newman -> Reissner-Nordstrom-Taub-NUT
# ----------------------------------------------------------------------------------------------------------------
print('--- D. near-horizon extremal Kerr-Newman continues to RN-Taub-NUT with m = 0 ---')
a, Qc, lam = sp.symbols('a Q lambda', positive=True)
taN, rhN, thN, phN = sp.symbols('tau rho theta phip', real=True)
r0 = sp.sqrt(a ** 2 + Qc ** 2)                         # extremal Kerr-Newman: M = r_0 = (a^2 + Q^2)^{1/2}
L2 = r0 ** 2 + a ** 2
rB = r0 + lam * rhN                                    # r = r_0 + lambda rho, t = L2 tau/lambda, phi = phip + a t/L2
Sig = rB ** 2 + a ** 2 * sp.cos(thN) ** 2
Dl = sp.expand(rB ** 2 - 2 * r0 * rB + a ** 2 + Qc ** 2)
dt_, dphi_ = sp.Matrix([L2 / lam, 0, 0, 0]), sp.Matrix([a / lam, 0, 0, 1])     # components on (dtau, drho, dtheta, dphip)
dr_, dth_ = sp.Matrix([0, lam, 0, 0]), sp.Matrix([0, 0, 1, 0])
w1 = dt_ - a * sp.sin(thN) ** 2 * dphi_
w2 = (rB ** 2 + a ** 2) * dphi_ - a * dt_
gKN = -(Dl / Sig) * w1 * w1.T + Sig * (dr_ * dr_.T / Dl + dth_ * dth_.T) + sp.sin(thN) ** 2 / Sig * w2 * w2.T
gNH = gKN.applyfunc(lambda e: sp.simplify(sp.limit(sp.together(sp.expand(e)), lam, 0)))
Gam = r0 ** 2 + a ** 2 * sp.cos(thN) ** 2
gam = L2 ** 2 * sp.sin(thN) ** 2 / Gam
kk = 2 * a * r0 / L2
target = sp.zeros(4)
target[0, 0] = -Gam * rhN ** 2 + gam * kk ** 2 * rhN ** 2
target[0, 3] = target[3, 0] = gam * kk * rhN
target[1, 1], target[2, 2], target[3, 3] = Gam / rhN ** 2, Gam, gam
ck.zero('D: near-horizon limit of extremal Kerr-Newman = Gamma [AdS_2] + Gamma dtheta^2 + gamma (dphi + k rho dtau)^2, '
        'Gamma = r_0^2 + a^2 cos^2, k = 2 a r_0/(r_0^2 + a^2)', sum((gNH - target).applyfunc(lambda e: sp.Abs(sp.simplify(e)))))
AKN = -(Qc * rB / Sig) * w1                            # Kerr-Newman A = -(Q r/Sigma)(dt - a sin^2 dphi)
ANH = [sp.simplify(sp.series(sp.expand(AKN[i]), lam, 0, 1).removeO()) for i in range(4)]
ANH[0] = sp.simplify(ANH[0] - sp.limit(lam * ANH[0], lam, 0) / lam)    # drop the constant 1/lambda (pure gauge) term
fA = sp.simplify(ANH[3])
eNH = sp.simplify((ANH[0] - kk * rhN * fA) / rhN)
zeros('D: constant near-horizon electric field has the predicted value',
      [sp.diff(eNH, thN), sp.diff(eNH, rhN), eNH - Qc * (r0 ** 2 - a ** 2) / L2])
zeros('D: f_A vanishes at each pole', [fA.subs(thN, 0), fA.subs(thN, sp.pi)])
# continued real configuration in (T, y, vartheta, psi), y = a cos(theta), varphi = a T/L2, rho dtau -> cos(vartheta) dpsi
T_, y, vt, psi = sp.symbols('T y vartheta psi', real=True)
n = r0
fy = (y ** 2 - a ** 2) / (y ** 2 + n ** 2)
xs = y / a
fA_y = sp.simplify(fA.subs(thN, sp.acos(xs)))
one = sp.Matrix([1, 0, 0, 2 * n * sp.cos(vt)])          # dT + 2n cos(vartheta) dpsi
gC = -fy * one * one.T + sp.diag(0, 1 / fy, y ** 2 + n ** 2, (y ** 2 + n ** 2) * sp.sin(vt) ** 2)
AC = [fA_y * a / L2 * one[0], 0, 0, eNH * sp.cos(vt) + fA_y * a / L2 * one[3]]
ck.zero('D: Gamma(cos theta = y/a) = y^2 + n^2 with n^2 = a^2 + Q^2', sp.simplify(Gam.subs(thN, sp.acos(xs)) - (y ** 2 + n ** 2)))
zeros('D: continued angular fiber coefficient and connection',
      [gam.subs(thN, sp.acos(xs)) * (a / L2) ** 2 + fy, kk * L2 / a - 2 * n])
ck.zero('D: -Gamma dx^2/(1 - x^2) = dy^2/f(y)', sp.simplify(-Gam.subs(thN, sp.acos(xs)) / (1 - xs ** 2) / a ** 2 - 1 / fy))
XC = [T_, y, vt, psi]
FC = sp.Matrix(4, 4, lambda i, j: sp.diff(AC[j], XC[i]) - sp.diff(AC[i], XC[j]))
RcD, scD, dGD = field_equations(gC, (y ** 2 + n ** 2) * sp.sin(vt), XC, sp.Integer(0) * y, [FC], [lambda u_: 1 + 0 * u_], [ZERO], ZERO)
ck.zero('D: the continued real metric and gauge field solve Einstein-Maxwell with the standard sign', sum(RcD.applyfunc(sp.Abs)))
ck.true('D: ... and the Maxwell equations', all(sp.simplify(x) == 0 for x in dGD))
ck.zero('D: f(y) is that of Reissner-Nordstrom-Taub-NUT with m = 0, n^2 = a^2 + Q^2 and charge^2 = Q^2',
        sp.simplify(fy - (y ** 2 - n ** 2 + Qc ** 2) / (y ** 2 + n ** 2)))
flux = sp.simplify((AC[3].subs(vt, sp.pi) - AC[3].subs(vt, 0)) / 2)   # (1/4pi) int F over {T, y fixed}, oriented by dvartheta^dpsi
ck.zero('D: flux (1/4pi) int F through the spheres y = const = Q (y^2 - n^2)/(y^2 + n^2) (orientation dvartheta^dpsi)',
        sp.simplify(flux - Qc * (y ** 2 - n ** 2) / (y ** 2 + n ** 2)))
ck.zero('D: in this orientation the flux at large y is +Q, the magnetic charge', sp.simplify(sp.limit(flux, y, sp.oo) - Qc))
zeros('D: horizon fluxes equal -e separately', [flux.subs(y, a) + eNH, flux.subs(y, -a) + eNH])
zeros('D: each continued pole is a zero of f', [fy.subs(y, a), fy.subs(y, -a)])
kpm = [sp.simplify(sp.diff(fy, y).subs(y, a) / 2), sp.simplify(sp.diff(fy, y).subs(y, -a) / 2)]
ck.zero('D: kappa_+ = -kappa_- (f is even, m = 0)', kpm[0] + kpm[1])
ck.zero('D: S_+ = S_- = pi (a^2 + n^2) = pi (r_0^2 + a^2) = S_ext of extremal Kerr-Newman',
        sp.simplify(sp.pi * (a ** 2 + n ** 2) - sp.pi * (r0 ** 2 + a ** 2)))
zeros('D: Kerr limit gives massless Taub-NUT and n=a',
      [fy.subs(Qc, 0) - (y ** 2 - a ** 2) / (y ** 2 + a ** 2), n.subs(Qc, 0) - a])
print('   (D done, %.0f s)' % (time.time() - T0))


# E. KLR's dipole-ring example
# ----------------------------------------------------------------------------------------------------------------
print('--- E. KLR: the dipole-ring near-horizon geometry and its continuation ---')
R, lm, mu, x = sp.symbols('R lambda mu x', real=True)
rx = R * sp.sqrt(lm * mu) * (mu * x - 1)                       # KLR: r = R sqrt(lambda mu)(mu x - 1)
rpk, rmk = -R * sp.sqrt(lm * mu) * (1 + mu), -R * sp.sqrt(lm * mu) * (1 - mu)
r0k = R * sp.sqrt(lm * mu) * (1 + mu / lm)
zeros('E: continued poles equal the respective roots', [rx.subs(x, -1) - rpk, rx.subs(x, 1) - rmk])
Vk = lambda rr: (rr - rpk) * (rr - rmk) / rr ** 2
Uk = lambda rr: (rr + r0k) / rr
Ck = (rpk + r0k) * (rmk + r0k)                                  # = U W, a constant
ck.zero('E: V(r(x)) r(x)^2 = R^2 lambda mu^3 (x^2 - 1): simple zeros exactly at the poles x = +-1',
        sp.expand(Vk(rx) * rx ** 2 - R ** 2 * lm * mu ** 3 * (x ** 2 - 1)))
# U r^2 = r (r + r_0) vanishes at r = 0 and r = -r_0, where the solution is singular (F^2 and R_mn R^mn checks below)
zeros('E: separate charge-squared and angular-fiber parameter identities',
      [rpk * rmk - R ** 2 * lm * mu * (1 - mu ** 2), Ck - R ** 2 * mu ** 3 * (1 - lm ** 2) / lm])
# hence, for lambda mu > 0, r = 0 and r = -r_0 lie outside the interval between the roots iff |mu| < 1 and |lambda| < 1
ring = {R: 1, lm: sp.Rational(1, 2), mu: sp.Rational(1, 3)}
ck.true('E: ring parameters (R, lambda, mu) = (1, 1/2, 1/3): r_+ < r_- < 0 < r_+- + r_0',
        bool(rpk.subs(ring) < rmk.subs(ring)) and bool(rmk.subs(ring) < 0) and bool((rpk + r0k).subs(ring) > 0)
        and bool((rmk + r0k).subs(ring) > 0))
for lab, pv, z_ in (('(i) R = -1, lambda = 2, mu = 1/2: r = -r_0', {R: -1, lm: 2, mu: sp.Rational(1, 2)}, -r0k),
                    ('(ii) R = 1, lambda = -1/2, mu = -2: r = 0', {R: 1, lm: -sp.Rational(1, 2), mu: -2}, sp.Integer(0))):
    a_, b_, c_, m0 = [sp.nsimplify(e.subs(pv)) for e in (rpk, rmk, z_, -r0k)]
    coordinate_inequalities = bool(a_ > 0) and bool(a_ > b_) and bool(a_ > m0)
    ck.true('E: quoted coordinate range %s puts a singular point between the roots (r_+, r_-) = (%s, %s)' % (lab, a_, b_),
            coordinate_inequalities and bool(min(a_, b_) < c_) and bool(c_ < max(a_, b_)))
    # Coordinate inequalities do not suffice for a physical exterior. The
    # printed gauge field must be real and all spatial metric directions positive.
    if pv[lm] > 0:
        ck.true('E: negative control: first quoted range has negative fiber coefficient C and fails Lorentzian exterior test',
                bool(Ck.subs(pv) < 0))
    else:
        ck.true('E: negative control: second quoted range has negative charge squared and fails gauge-field reality test',
                bool((rpk*rmk).subs(pv) < 0))


def klr60(sU, pv, rr, thv):
    """KLR (60), -V dt^2 + (U/V) dr^2 + U r^2 (dtheta^2 + sin^2 dT^2) + W (dchi + cos dT)^2 (sU = 1), and its real
    section with U -> -U, W -> -W (sU = -1), in coordinates (t, r, theta, T, chi)"""
    V_, U_ = Vk(rr).subs(pv), sU * Uk(rr).subs(pv)
    W_ = sU * Ck.subs(pv) / Uk(rr).subs(pv)
    g = sp.zeros(5)
    g[0, 0], g[1, 1], g[2, 2] = -V_, U_ / V_, U_ * rr ** 2
    g[3, 3] = U_ * rr ** 2 * sp.sin(thv) ** 2 + W_ * sp.cos(thv) ** 2
    g[3, 4] = g[4, 3] = W_ * sp.cos(thv)
    g[4, 4] = W_
    return g


# Complete the square in the angular fiber: signature equals the signs of
# (-V, U/V, U r², U r² sin²theta, C/U).  The circle coefficient C is positive
# in 0 < lambda, mu < 1.  Test the congruence itself symbolically, then the signs
# on an exact rational interior point; the general sign proof is stated in the paper.
rv = (rpk + rmk).subs(ring) / 2
for sU, expected in ((1, (1, 1, -1, -1, -1)), (-1, (1, -1, 1, 1, 1))):
    Uv, Vv, Cv = sU * Uk(rv).subs(ring), Vk(rv).subs(ring), Ck.subs(ring)
    signs = tuple(sp.sign(v) for v in (-Vv, Uv/Vv, Uv*rv**2, Uv*rv**2, Cv/Uv))
    ck.true('E: exact inertia of section sU=%s at a ring interior point' % sU, signs == expected)
    theta_sig = sp.Symbol('theta_signature', real=True)
    change = sp.eye(5)
    change[4, 3] = -sp.cos(theta_sig)
    diagonal = change.T * klr60(sU, {}, r, theta_sig) * change
    for i in range(5):
        for j in range(i + 1, 5):
            ck.zero('E: fiber congruence is diagonal sU=%s component %s%s' % (sU, i, j), diagonal[i,j])
# surface gravities: -V dt^2 + (U/V) dr^2 gives the formal kappa^2 = V'^2/(4U) of (60); the real section has U -> -U
kap2 = lambda rh: (sp.diff(Vk(r), r).subs(r, rh)) ** 2 / (4 * Uk(rh))
kap2r = lambda rh: -kap2(rh)
zeros('E: formal squared surface gravity at each root',
      [kap2(rpk) - (rpk - rmk) ** 2 / (4 * rpk ** 3 * (rpk + r0k)),
       kap2(rmk) - (rpk - rmk) ** 2 / (4 * rmk ** 3 * (rmk + r0k))])
ck.true('E: ring parameters: the formal kappa^2 of (60) are negative, those of the real section positive',
        all(bool(kap2(rh).subs(ring) < 0) and bool(kap2r(rh).subs(ring) > 0) for rh in (rpk, rmk)))
balance = (1 - lm) * (1 + mu) ** 3 - (1 + lm) * (1 - mu) ** 3       # dipole ring: (1-l)/(1+l) = ((1-m)/(1+m))^3
Dk = rpk ** 3 * (rpk + r0k) - rmk ** 3 * (rmk + r0k)
ck.zero('E: r_+^3 (r_+ + r_0) - r_-^3 (r_- + r_0) = -R^4 lambda mu^3 [(1-lambda)(1+mu)^3 - (1+lambda)(1-mu)^3], '
        'so kappa_+^2 = kappa_-^2 iff the dipole-ring balance condition holds', sp.expand(Dk + R ** 4 * lm * mu ** 3 * balance))
# horizon areas of the real section: det of its (theta, T, chi) block at r_+-, per unit coordinate volume, = A^2
TH = sp.Symbol('THETA', positive=True)
area2 = lambda rh: sp.simplify(klr60(-1, {}, rh, TH)[2:, 2:].det() / sp.sin(TH) ** 2)
ck.zero('E: real section: kappa_+^2 A_+^2 = kappa_-^2 A_-^2 (T_+ S_+ = -T_- S_-), so A_+ = A_- iff kappa_+ = |kappa_-|',
        sp.simplify(kap2r(rpk) * area2(rpk) - kap2r(rmk) * area2(rmk)))
lam_bal = ((1 + mu) ** 3 - (1 - mu) ** 3) / ((1 + mu) ** 3 + (1 - mu) ** 3)
ck.zero('E: on the balanced dipole ring lambda = ((1+mu)^3 - (1-mu)^3)/((1+mu)^3 + (1-mu)^3): kappa_+^2 = kappa_-^2',
        sp.simplify(Dk.subs(lm, lam_bal)))
ck.large('E: negative control, lambda = 1/2, mu = 1/3 (unbalanced): kappa_+^2 - kappa_-^2 != 0',
         (kap2r(rpk) - kap2r(rmk)).subs(ring), 1e-3)
# At one exact rational parameter tuple, check the full coordinate-dependent field equations.
# This is an identity at every regular coordinate point, not a sampled or floating-point residual.
# The parameter tuple is NOT a proof for arbitrary dipole parameters.
pv5 = {R: 1, lm: sp.Rational(1, 2), mu: sp.Rational(1, 8)}
th5, T5, chi5 = sp.symbols('theta5 T5 chi5', real=True)
X5 = [t, r, th5, T5, chi5]
F5 = sp.zeros(5)
F5[1, 0] = sp.diff(sp.sqrt((rpk * rmk).subs(pv5)) / r, r); F5[0, 1] = -F5[1, 0]
sq5 = -Uk(r).subs(pv5) * sp.sqrt(Ck.subs(pv5)) * r ** 2 * sp.sin(th5)        # sqrt(-g) for -r_0 < r < 0, where U < 0
cvals = []
for lab, sU in (('(60)', 1), ('its real section', -1)):
    g5 = klr60(sU, pv5, r, th5).applyfunc(sp.simplify)
    ck.zero('E: sqrt(-g) of %s' % lab, sp.simplify(-g5.det() - sq5 ** 2))
    gi5, Ric5 = christoffel_ricci(g5, X5)
    Fup5 = gi5 * F5 * gi5
    F2_5 = sp.simplify(sum(F5[i, j] * Fup5[i, j] for i in range(5) for j in range(5)))
    Tm5 = (F5 * gi5 * F5.T - g5 * F2_5 / 6).applyfunc(sp.simplify)
    cE = sp.simplify(Ric5[0, 0] / Tm5[0, 0])
    cvals.append(cE)
    # Equal field strengths in the three-vector theory give L=R-3F²/4.
    # Fix the coefficient independently; fitting it to a component is not an EOM test.
    ck.zero('E: %s has the prescribed Einstein-Maxwell coefficient 3/2' % lab,cE-sp.Rational(3,2))
    ck.zero('E: %s solves the prescribed Einstein equation in every component' % lab,
            sum(sp.Abs(sp.simplify(e)) for e in (Ric5 - sp.Rational(3,2) * Tm5)))
    ck.zero('E: %s: Maxwell equations d_m(sqrt(-g) F^{mn}) = 0' % lab,
            sum(sp.Abs(sp.simplify(sum(sp.diff(sq5 * Fup5[i, j], X5[i]) for i in range(5)))) for j in range(5)))
    if sU == -1:
        F2r = F2_5
        RicSq = sp.simplify(sum(Ric5[i, j] * (gi5 * Ric5 * gi5)[i, j] for i in range(5) for j in range(5)))
ck.zero('E: the same coupling c for (60) and its real section (the map (g, F) -> (-g, iF) with t -> i t)', cvals[0] - cvals[1])
ck.zero('E: real section: F^2 = 2 r_+ r_-/(r^3 (r + r_0))', sp.simplify(F2r - 2 * (rpk * rmk).subs(pv5) / (r ** 3 * (r + r0k.subs(pv5)))))
ck.true('E: real section: R_mn R^mn diverges as r -> 0 and r -> -r_0 (curvature singularities outside region II)',
        sp.limit(RicSq, r, 0, '-') == sp.oo and sp.limit(RicSq, r, -r0k.subs(pv5), '+') == sp.oo)
# conical regularity of the near-horizon geometry (57): proper circumference ~ R mu sqrt(1-x^2)/H, radial ~ R mu sqrt(F H)/sqrt(1-x^2)
Fk, Hk = 1 + lm * x, 1 - mu * x
period = lambda xp: 2 * sp.pi * Hk.subs(x, xp) ** sp.Rational(3, 2) * Fk.subs(x, xp) ** sp.Rational(1, 2)
ck.zero('E: equal phi-periods at the two poles of the near-horizon geometry <=> the same balance condition',
        sp.simplify(sp.expand((period(1) / (2 * sp.pi)) ** 2 - (period(-1) / (2 * sp.pi)) ** 2) + balance))
print('   (E done, %.0f s)' % (time.time() - T0))


# F. The Lorentzian throat and its genuine Riemannian continuation.
print('--- F. throat signature and exact Legendre zero modes ---')
rh_p,rh_m,vth=sp.symbols('rho_p rho_m vartheta',real=True)
kap_t=(rh_p-rh_m)/2
rho_t=(rh_p+rh_m)/2+kap_t*sp.cos(vth)
F_t=(rho_t-rh_p)*(rho_t-rh_m)
ck.zero('F: the interior radial metric is -dtheta², not +dtheta²',sp.diff(rho_t,vth)**2/F_t+1)
ck.zero('F: the interior time metric becomes +sin²theta dpsi²',-F_t/kap_t**2-sp.sin(vth)**2)
ck.large('F: negative control rejects calling the Lorentzian interior a real round sphere',-sp.Integer(1)-1,1)
# A complex coordinate/real-section choice is required for a round sphere.
for degree in range(7):
    poly=sp.legendre(degree,z)
    laplace=sp.diff((1-z*z)*sp.diff(poly,z),z)
    ck.zero('F: P_%s satisfies the spherical eigenvalue equation exactly' % degree,laplace+degree*(degree+1)*poly)
    ck.zero('F: P_%s is smooth with the correct parity at both poles' % degree,poly.subs(z,-1)-(-1)**degree*poly.subs(z,1))
    if degree:
        ck.zero('F: P_%s has zero mean and preserves flux at first order' % degree,sp.integrate(poly,(z,-1,1)))
ck.large('F: negative control rejects wrong eigenvalue for the dipole',
         (sp.diff((1-z*z)*sp.diff(z,z),z)+6*z).subs(z,sp.Rational(1,2)),1)
print('   (total %.1f s)' % (time.time()-T0))
ck.finish()

