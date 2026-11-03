r"""
toda_exact.py -- the exact A1+A1 family (alpha1 alpha2 = -4) in closed form, for all couplings on that curve, all
temperatures and all moduli (Section 5.4 and Appendix B.4): a symbolic replacement for the numerical A1+A1 rows of
Table 1.

Model (eq:action) with one scalar, V = 0, V_eff = Q1^2 e^{alpha1 phi} + Q2^2 e^{alpha2 phi}, alpha1 alpha2 = -4.
With w_i = (2, alpha_i) and |w_i|^2 = 4 + alpha_i^2, the vectors w_1, w_2 are orthogonal, and the radial Lagrangian
L = U'^2 + phi'^2 + e^{2U} V_eff (d tau = dr/(a^2 b^2)) splits into two Liouville systems.  In the gauge
a^2 b^2 = (r - r+)(r - r-), with c = (r+ - r-)/2 and the LINEAR functions
    l_i(r) = [(r+ - r) e^{-c tau_i} + (r - r-) e^{c tau_i}]/2 ,
the general solution with horizons at r+- is
    b^2 = prod_i [ |w_i|^2 Q_i^2 l_i^2/(4 c^2) ]^{4/|w_i|^2},      phi = phi_0 - 2 sum_i (alpha_i/|w_i|^2) ln l_i ,
    phi_0 = sum_i (alpha_i/|w_i|^2) ln(4 c^2/(|w_i|^2 Q_i^2)) ,
for arbitrary tau_1, tau_2.  The key identity is  Q_i^2 e^{alpha_i phi}/b^2 = 4 c^2/(|w_i|^2 l_i^2), which makes all
reduced equations rational in r.  Checked here, symbolically in alpha_1 (alpha_2 = -4/alpha_1), tau_i, r+-, Q_i:
  * that identity, and (red1)-(red4);
  * l_i > 0 on [r-, r+] (linear, with positive endpoint values c e^{+-c tau_i}): every member has a regular inner
    horizon, for every amplitude (type (b) of Section 5.4);
  * b+^2 b-^2 = V_*^2 with V_* = min V_eff = prod_i (|w_i|^2 Q_i^2/4)^{4/|w_i|^2}: S+ S- = S_ext^2 (Theorem 5.3);
  * the tau-shift tau_i -> tau_i + tau_0 keeps b+ b- and phi_+- (the symmetry of Section 5.4);
  * b+^2/b-^2 = exp(16 c sum_i tau_i/|w_i|^2): the balanced member of Section 8 (b+ = b-) is sum_i tau_i/|w_i|^2 = 0,
    i.e. 4 tau_1 + alpha_1^2 tau_2 = 0 (tau_1 + tau_2 = 0 only for alpha_1 = 2);
  * phi_- - phi_* = -(phi_+ - phi_*) exactly: the relation (-1)^nu of Section 5.4 holds non-linearly (nu = 1);
  * the potential differences (Phi_- - Phi_+)/(r+ - r-) = (1/2) d ln S_ext/d Q_1 and its magnetic analogue;
  * at linear order about tau_1 = tau_2, delta phi is proportional to P_1(z), the regular Legendre mode with nu = 1,
    also when the tau-shift that keeps b_+ fixed is included (delta phi = -c P_1(z) for alpha_1 = 2);
  * the AGM relation (phi_+- - phi_*) z_inf/(phi_inf - phi_*) -> +-1 (AGM (119)-(120)), in closed form.
Domain: c > 0, Q_i > 0, alpha_1 > 0, X_i = exp(c tau_i) > 0, and r in [r_-, r_+].
The labels may be exchanged to cover alpha_1 < 0. Logarithms and noninteger powers below are real on this
positive strip; expand_log(force=True) is used only with that domain. Arbitrary tau_i do NOT guarantee an
asymptotically flat exterior. X_i > 1 suffices for positive affine factors on r >= r_+; the asymptotic AGM
comparison is restricted to that exterior domain. No limiting c = 0 geometry is claimed by this parametrization.
Each residual is checked separately; no sum of unrelated residuals can hide a cancellation.
Run:  python3 toda_exact.py   (well under a minute)
"""
import sympy as sp
import checklib as ck

r = sp.Symbol('r', real=True)
rm, Q1, Q2, al1, c = sp.symbols('r_m Q1 Q2 alpha1 c', positive=True)
rp = rm + 2 * c  # enforce a strictly positive horizon separation
t1, t2, t0, tau, eps_ = sp.symbols('tau1 tau2 tau0 tau epsilon', real=True)
X1, X2 = sp.symbols('X1 X2', positive=True)           # X_i = e^{c tau_i}: makes the reduced equations rational
import time
T0 = time.time()
u = sp.Symbol('u')
al = (al1, -4 / al1)
w2 = tuple(sp.simplify(4 + a_ ** 2) for a_ in al)
Qs, taus = (Q1, Q2), (t1, t2)
Xs = (X1, X2)                                          # X_i = e^{c tau_i}
l = [((rp - r) / Xi + (r - rm) * Xi) / 2 for Xi in Xs]
K = sp.Mul(*[(w2[i] * Qs[i] ** 2 / (4 * c ** 2)) ** (4 / w2[i]) for i in range(2)])
p = [8 / w2[i] for i in range(2)]                       # b^2 = K l_1^{p_1} l_2^{p_2}
b2 = K * l[0] ** p[0] * l[1] ** p[1]
phi0 = sum(al[i] / w2[i] * sp.log(4 * c ** 2 / (w2[i] * Qs[i] ** 2)) for i in range(2))
ph = phi0 - 2 * sum(al[i] / w2[i] * sp.log(l[i]) for i in range(2))
P = (r - rp) * (r - rm)

print('--- A1+A1: alpha_2 = -4/alpha_1, |w_i|^2 =', w2)
ck.zero('sum 4/|w_i|^2 = 1', 4 / w2[0] + 4 / w2[1] - 1)
ck.zero('sum alpha_i/|w_i|^2 = 0', al[0] / w2[0] + al[1] / w2[1])
ck.zero('Liouville exponent vectors are orthogonal', 4 + al[0] * al[1])
# key identity, by exponent arithmetic on logarithms
for i in range(2):
    lhs = sp.log(Qs[i] ** 2) + al[i] * ph - sp.log(b2)
    rhs = sp.log(4 * c ** 2 / (w2[i] * l[i] ** 2))
    ck.zero('Q_%d^2 e^{alpha_%d phi}/b^2 = 4c^2/(|w_%d|^2 l_%d^2)' % (i + 1, i + 1, i + 1, i + 1),
            sp.simplify(sp.expand_log(lhs - rhs, force=True)))
Vb = 4 * c ** 2 * sum(1 / (w2[i] * l[i] ** 2) for i in range(2))              # V_eff/b^2
dVb = 4 * c ** 2 * sum(al[i] / (w2[i] * l[i] ** 2) for i in range(2))         # (dV_eff/dphi)/b^2
D = sp.together(sum(p[i] * sp.diff(l[i], r) / l[i] for i in range(2)))        # (b^2)'/b^2
dph = sp.diff(ph, r)
d = lambda e: sp.diff(e, r)
reds = {'b"/b + phi\'^2 = D\'/2 + D^2/4 + phi\'^2 = 0 (red1)': d(D) / 2 + D ** 2 / 4 + dph ** 2,
        '(red2): (P phi\')\' = (dV/dphi)/(2 b^2)': d(P * dph) - dVb / 2,
        '(red3) tt: (P\' - P D)\'/2 = V/b^2': d(P.diff(r) - P * D) / 2 - Vb,
        '(red3) theta theta: -(P D)\'/2 = V/b^2 - 1': -d(P * D) / 2 - Vb + 1,
        '(red4): -1 + P D^2/4 + (P\' D - P D^2)/2 + V/b^2 - P phi\'^2 = 0': -1 + P * D ** 2 / 4 + (P.diff(r) * D - P * D ** 2) / 2 + Vb - P * dph ** 2}
for lab, e in reds.items():
    eX = e
    ck.zero('A1+A1, all alpha_1, tau_1, tau_2, r_+-: %s' % lab, sp.cancel(sp.together(eX)))
    print('      (%.1f s)' % (time.time() - T0), flush=True)
ck.zero('a^2 b^2 = P is consistent with (a^2 b^2)" = 2 (red1)', sp.diff(P, r, 2) - 2)
lp = [li.subs(r, rp) for li in l]
lmn = [li.subs(r, rm) for li in l]
for i in range(2):
    ck.zero('l_%d(r_+) = c X_%d > 0' % (i + 1, i + 1), lp[i] - c * Xs[i])
    ck.zero('l_%d(r_-) = c/X_%d > 0' % (i + 1, i + 1), lmn[i] - c / Xs[i])
    ck.zero('l_%d is affine, hence positive between its positive endpoints' % (i + 1), sp.diff(l[i], r, 2))
Vs = sp.Mul(*[(w2[i] * Qs[i] ** 2 / 4) ** (4 / w2[i]) for i in range(2)])
Veff = Q1 ** 2 * sp.exp(al[0] * u) + Q2 ** 2 * sp.exp(al[1] * u)
ustar = sp.log(-al[1] * Q2 ** 2 / (al[0] * Q1 ** 2)) / (al[0] - al[1])
ck.zero('u_* = ln(-alpha_2 Q_2^2/(alpha_1 Q_1^2))/(alpha_1 - alpha_2) is the critical point of V_eff',
        sp.simplify(sp.diff(Veff, u).subs(u, ustar)))
ck.zero('V_* = V_eff(u_*) = prod_i (|w_i|^2 Q_i^2/4)^{4/|w_i|^2}',
        sp.simplify(sp.expand_log(sp.log(sp.simplify(Veff.subs(u, ustar))) - sp.log(Vs), force=True)))
lnbb = sp.expand_log(sp.log(b2.subs(r, rp)) + sp.log(b2.subs(r, rm)), force=True)
ck.zero('b_+^2 b_-^2 = V_*^2 for all tau_1, tau_2: S_+ S_- = S_ext^2', sp.simplify(lnbb - 2 * sp.expand_log(sp.log(Vs), force=True)))
lam_ = sp.Symbol('Lambda', positive=True)
shift = {X1: lam_ * X1, X2: lam_ * X2}                 # tau_i -> tau_i + tau_0, Lambda = e^{c tau_0}
for name, horizon in (('+', rp), ('-', rm)):
    ck.zero('common tau-shift leaves phi_%s unchanged' % name,
            sp.expand_log(ph.subs(shift).subs(r, horizon) - ph.subs(r, horizon), force=True))
shifted_product_log = sp.expand_log(lnbb.subs(shift) - lnbb, force=True)
ck.zero('common tau-shift leaves the horizon area product unchanged', shifted_product_log)
lnratio = sp.expand_log(sp.log(b2.subs(r, rp)) - sp.log(b2.subs(r, rm)), force=True)
ck.zero('b_+^2/b_-^2 = prod_i X_i^{16/|w_i|^2} = exp(16 c sum_i tau_i/|w_i|^2)',
        sp.simplify(lnratio - 16 * sum(sp.log(Xs[i]) / w2[i] for i in range(2))))
ck.zero('so b_+ = b_- (the balanced member) iff sum_i tau_i/|w_i|^2 = (4 tau_1 + alpha_1^2 tau_2)/(4 |w_1|^2) = 0',
        sp.simplify(t1 / w2[0] + t2 / w2[1] - (4 * t1 + al1 ** 2 * t2) / (4 * w2[0])))
ck.large('negative control: for alpha_1 = 1, tau_1 + tau_2 = 0 is not balanced (tau_1 = 1/2)',
         lnratio.subs({X1: sp.exp(c / 2), X2: sp.exp(-c / 2)}).subs({al1: 1, c: sp.Rational(13, 40), rm: sp.Rational(7, 20), Q1: 1, Q2: 1}), 1e-2)
php = sp.simplify(sp.expand_log(ph.subs(r, rp), force=True))
phm = sp.simplify(sp.expand_log(ph.subs(r, rm), force=True))
phs = sp.simplify(sp.expand_log(ph.subs({X1: 1, X2: 1}).subs(r, rp), force=True))
ck.zero('tau_1 = tau_2: phi is constant (the doubly-extremal solution) ...', sp.simplify(sp.diff(ph.subs(X2, X1), r)))
ck.zero('... and equal to u_*, the attractor value', sp.simplify(sp.expand_log(phs - ustar, force=True)))
ck.zero('phi_- - phi_* = -(phi_+ - phi_*) for all tau_i: the relation (-1)^nu, nu = 1, holds non-linearly',
        sp.simplify(php + phm - 2 * phs))
ck.large('negative control: phi_+ - phi_* != 0 for tau_1 != tau_2',
         (php - phs).subs({X1: sp.Rational(3, 2), X2: 1, c: sp.Rational(13, 40), rm: sp.Rational(7, 20), al1: 1, Q1: 1, Q2: 1}), 1e-3)
# potential differences: E_1 = Q_1 e^{alpha_1 phi}/b^2 (f_1 = e^{-alpha_1 phi}), dual: f_2 Q_2/b^2 = Q_2 e^{alpha_2 phi}/b^2
for i, lab in ((0, '(Phi_- - Phi_+)'), (1, '(Psi_- - Psi_+)')):
    integrand = 4 * c ** 2 / (w2[i] * Qs[i] * l[i] ** 2)
    # This primitive remains regular at X_i = 1, unlike a formula dividing by l_i'.
    primitive = 4 * c ** 2 / (w2[i] * Qs[i]) * (r - rm) / (lmn[i] * l[i])
    ck.zero('potential %d: differentiate the exact primitive' % (i + 1), sp.cancel(sp.diff(primitive, r) - integrand))
    I = sp.simplify(primitive.subs(r, rp) - primitive.subs(r, rm))
    ck.zero('potential %d: X_%d = 1 is included without a limiting prescription' % (i + 1, i + 1),
            I.subs(Xs[i], 1) - 8 * c / (w2[i] * Qs[i]))
    ck.zero('%s/(r_+ - r_-) = (1/2) d ln S_ext/d Q_%d = 4/(|w_%d|^2 Q_%d), independent of tau_i (moduli and temperature)'
            % (lab, i + 1, i + 1, i + 1), sp.simplify(I / (rp - rm) - sp.diff(sp.expand_log(sp.log(Vs), force=True), Qs[i]) / 2))

# linear order about tau_1 = tau_2 for general alpha_1: delta phi is proportional to P_1(z)
tau, eps_ = sp.symbols('tau epsilon', real=True)
ellg = ((rp - r) * sp.exp(-c * tau) + (r - rm) * sp.exp(c * tau)) / 2
sg = sp.simplify(sp.solve(ellg, r)[0])
zg = ((rp - sg + rm - sg) * (r - sg) - 2 * (rp - sg) * (rm - sg)) / ((rp - rm) * (r - sg))
dphig = sp.diff(ph.subs({X1: sp.exp(c * (tau + eps_)), X2: sp.exp(c * (tau - eps_))}), eps_).subs(eps_, 0)
ck.zero('general alpha_1: for tau_1,2 = tau +- eps, delta phi = -4 c (alpha_1/|w_1|^2) P_1(z) eps, the regular Legendre mode with nu = 1',
        sp.simplify(dphig + 4 * c * al[0] / w2[0] * zg))
# this variation moves b_+ unless alpha_1 = 2; adding the tau-shift tau_0 = kk eps that keeps b_+ fixed does not change delta phi
kk = -4 * (1 / w2[0] - 1 / w2[1])
pert = {X1: sp.exp(c * (tau + eps_ + kk * eps_)), X2: sp.exp(c * (tau - eps_ + kk * eps_))}
dlnbp = sp.diff(sp.expand_log(sp.log(b2.subs(r, rp).subs(pert)), force=True), eps_).subs(eps_, 0)
ck.zero('... with the compensating tau-shift tau_0 = -4 (1/|w_1|^2 - 1/|w_2|^2) eps, b_+ is fixed at this order', sp.simplify(dlnbp))
ck.zero('... and delta phi is unchanged', sp.simplify(sp.diff(ph.subs(pert), eps_).subs(eps_, 0) - dphig))
# linear order about the doubly-extremal member, and the AGM relation, symmetric case alpha_1 = 2
sym = {al1: 2}
phS = ph.subs(sym)
ell = ((rp - r) * sp.exp(-c * tau) + (r - rm) * sp.exp(c * tau)) / 2         # tau_1 = tau_2 = tau: b is proportional to ell
toT = lambda e: e.subs({X1: sp.exp(c * (tau + eps_)), X2: sp.exp(c * (tau - eps_))})
s_ = sp.simplify(sp.solve(ell, r)[0])                                          # b = 0 at r = s (b = r - s up to scale)
rt = lambda x: x - s_
z_t = ((rt(rp) + rt(rm)) * rt(r) - 2 * rt(rp) * rt(rm)) / ((rp - rm) * rt(r))  # z of Section 5.4 in the gauge b ~ r - s
dphi = sp.simplify(sp.diff(toT(phS), eps_).subs(eps_, 0))
ck.zero('alpha_1 = 2, linear order at fixed b_+ (tau_1,2 = tau +- eps): delta phi = -c P_1(z), the regular Legendre mode with nu = 1',
        sp.simplify(dphi + c * z_t))
zinf = sp.simplify(sp.limit(z_t, r, sp.oo))
ck.zero('z_inf = coth(c tau)', sp.simplify(sp.expand((zinf - sp.cosh(c * tau) / sp.sinh(c * tau)).rewrite(sp.exp))))
fam = toT(phS)
php_, phm_ = sp.simplify(fam.subs(r, rp)), sp.simplify(fam.subs(r, rm))
# at large r, l_i = r (X_i - 1/X_i)/2 + O(1) and the ln r terms cancel (sum alpha_i/|w_i|^2 = 0); X_i > 1 for a flat exterior
phinf_X = phi0 - 2 * sum(al[i] / w2[i] * sp.log((Xs[i] - 1 / Xs[i]) / 2) for i in range(2))
ck.zero('all couplings: the asymptotic ln r coefficient vanishes', sum(al[i] / w2[i] for i in range(2)))
# For X_i > 1, each positive affine factor has l_i = r (X_i - 1/X_i)/2 + O(1),
# so the preceding identity proves phi - phi_inf = O(1/r) for the whole family.
dphiinf_general = sp.diff(toT(phinf_X), eps_).subs(eps_, 0)
for name, horizon, sign in (('+', rp, 1), ('-', rm, -1)):
    ck.zero('all A1+A1 couplings: linear AGM ratio at horizon %s is %s' % (name, sign),
            sp.simplify((dphig.subs(r, horizon) * zinf - sign * dphiinf_general).rewrite(sp.exp)))
phinf = toT(phinf_X.subs(sym))
phst = sp.simplify(fam.subs(eps_, 0).subs(r, rp))
agm_p = (php_ - phst) * zinf / (phinf - phst)
agm_m = (phm_ - phst) * zinf / (phinf - phst)
lim_p, lim_m = sp.simplify(sp.limit(agm_p, eps_, 0)), sp.simplify(sp.limit(agm_m, eps_, 0))
print('   AGM ratios as phi_inf -> phi_*:', lim_p, lim_m)
ck.zero('AGM (119)-(120): outer linear ratio is +1', lim_p - 1)
ck.zero('AGM (119)-(120): inner linear ratio is -1', lim_m + 1)
ck.large('AGM: at finite amplitude (eps = 0.3, c tau = 0.5) the ratio differs from 1 (a linear-order statement)',
         sp.N(agm_p.subs({eps_: sp.Rational(3, 10), tau: sp.Rational(1, 2) / c}).subs({c: sp.Rational(13, 40), rm: sp.Rational(7, 20), Q1: 1, Q2: 1}) - 1, 20), 1e-4)
# Wrong stress normalization and wrong potential sign must not pass the same equations.
point = {r: sp.Rational(1, 2), c: sp.Rational(13, 40), rm: sp.Rational(7, 20),
         al1: 1, X1: sp.Rational(3, 2), X2: 2, Q1: 1, Q2: 1}
ck.large('negative control: removing scalar stress violates b equation',
         (d(D) / 2 + D ** 2 / 4).subs(point), sp.Rational(1, 1000))
ck.large('negative control: reversing the scalar source violates its equation',
         (d(P * dph) + dVb / 2).subs(point), sp.Rational(1, 1000))
ck.finish()
