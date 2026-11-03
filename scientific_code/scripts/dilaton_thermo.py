r"""
dilaton_thermo.py -- horizon thermodynamics of the exact dilaton family (see dilaton_exact.py)
and numerical verification (50 digits) of:
  (i)   first laws at r+ and r-  (signed T_-):  dM = T dS + Phi^a dQ_a - Sigma dphi0
  (ii)  T+S+ = -T-S- = Delta/2
  (iii) S+S- = (pi Veff_*)^2 = (2 pi Q1 Q2)^2, independent of (M, phi0)
  (iv)  Smarr at both horizons: M = 2 T S + Phi^a Q_a
  (v)   region-II integrals: int (Veff/B^2 - 1) dr = 0, int dVeff/dphi / B^2 dr = 0,
        Komar: T+S+ - T-S- = int Veff/(2B^2) dr
  (vi)  mixed boundary condition Sigma = W'(phi_inf), W = w phi^2/2: d(M + W) = T+ dS+ along that slice
The radial Noether function K(r) itself is evaluated on this family in dilaton_exact.py.
"""
import mpmath as mp
import checklib as ck
mp.mp.dps = 50


def data(M, p0, Q1, Q2):
    S = (mp.e ** (-2 * p0) * Q2 ** 2 - mp.e ** (2 * p0) * Q1 ** 2) / (2 * M)
    V = mp.e ** (2 * p0) * Q1 ** 2 + mp.e ** (-2 * p0) * Q2 ** 2
    D = mp.sqrt(M ** 2 + S ** 2 - V)
    rp, rm = M + D, M - D
    B2 = lambda r: r ** 2 - S ** 2
    A = lambda r: (r - rp) * (r - rm) / B2(r)
    phi = lambda r: p0 + mp.log((r + S) / (r - S)) / 2
    Veff = lambda r: mp.e ** (2 * phi(r)) * Q1 ** 2 + mp.e ** (-2 * phi(r)) * Q2 ** 2
    dVeff = lambda r: 2 * mp.e ** (2 * phi(r)) * Q1 ** 2 - 2 * mp.e ** (-2 * phi(r)) * Q2 ** 2
    At1 = lambda r: -Q1 * mp.e ** (2 * p0) / (r - S)
    At2 = lambda r: -Q2 * mp.e ** (-2 * p0) / (r + S)
    out = {}
    for lab, rh in (('+', rp), ('-', rm)):
        T = mp.diff(A, rh) / (4 * mp.pi)
        Sbh = mp.pi * B2(rh)
        out[lab] = dict(r=rh, T=T, S=Sbh, Phi1=-At1(rh), Phi2=-At2(rh))
    out['Sigma'] = S; out['Delta'] = D; out['M'] = M
    out['int1'] = mp.quad(lambda r: Veff(r) / B2(r) - 1, [rm, rp])
    out['int2'] = mp.quad(lambda r: dVeff(r) / B2(r), [rm, rp])
    out['komar'] = mp.quad(lambda r: Veff(r) / (2 * B2(r)), [rm, rp])
    return out


def check(M, p0, Q1, Q2, h=mp.mpf('1e-20')):
    d0 = data(M, p0, Q1, Q2)
    print('--- M=%s phi0=%s Q1=%s Q2=%s : r+=%s r-=%s Sigma=%s' % (M, p0, Q1, Q2, mp.nstr(d0['+']['r'], 8), mp.nstr(d0['-']['r'], 8), mp.nstr(d0['Sigma'], 8)))
    # derivatives by central differences in each parameter
    pars = [M, p0, Q1, Q2]
    for i, name in enumerate(['M', 'phi0', 'Q1', 'Q2']):
        pp = list(pars); pm = list(pars)
        pp[i] += h; pm[i] -= h
        dp, dm = data(*pp), data(*pm)
        dM = (dp['M'] - dm['M']) / (2 * h)
        dphi0 = 1 if i == 1 else 0
        dQ1 = 1 if i == 2 else 0
        dQ2 = 1 if i == 3 else 0
        for lab in '+-':
            dS = (dp[lab]['S'] - dm[lab]['S']) / (2 * h)
            lhs = dM
            rhs = d0[lab]['T'] * dS + d0[lab]['Phi1'] * dQ1 + d0[lab]['Phi2'] * dQ2 - d0['Sigma'] * dphi0
            print('   first law at r%s, variation %-4s: residual = %s' % (lab, name, mp.nstr(lhs - rhs, 5)))
            ck.small('first law at r%s, d/d%s' % (lab, name), lhs - rhs, 1e-25)
            if name == 'phi0' and lab == '+':
                ck.large('negative control: first law at r+ without the scalar-charge term, d/dphi0',
                         lhs - (rhs + d0['Sigma'] * dphi0), 1e-2)
    Tp, Sp, Tm, Sm = d0['+']['T'], d0['+']['S'], d0['-']['T'], d0['-']['S']
    print('   T+S+ + T-S- =', mp.nstr(Tp * Sp + Tm * Sm, 5), ';  T+S+ - Delta/2 =', mp.nstr(Tp * Sp - d0['Delta'] / 2, 5))
    print('   S+S- - (2 pi Q1 Q2)^2 =', mp.nstr(Sp * Sm - (2 * mp.pi * Q1 * Q2) ** 2, 5))
    ck.small('T+S+ + T-S- = 0', Tp * Sp + Tm * Sm, 1e-40)
    ck.small('T+S+ = Delta/2', Tp * Sp - d0['Delta'] / 2, 1e-40)
    ck.small('S+S- = (2 pi Q1 Q2)^2', Sp * Sm - (2 * mp.pi * Q1 * Q2) ** 2, 1e-40)
    for lab in '+-':
        dd = d0[lab]
        sm = M - 2 * dd['T'] * dd['S'] - dd['Phi1'] * Q1 - dd['Phi2'] * Q2
        print('   Smarr at r%s: M - 2TS - Phi.Q =' % lab, mp.nstr(sm, 5))
        ck.small('Smarr relation at r%s' % lab, sm, 1e-40)
    print('   int (Veff/B^2 - 1) =', mp.nstr(d0['int1'], 5), ';  int dVeff/B^2 =', mp.nstr(d0['int2'], 5),
          ';  Komar: T+S+ - T-S- - int Veff/2B^2 =', mp.nstr(Tp * Sp - Tm * Sm - d0['komar'], 5))
    ck.small('averaged equation int (Veff/B^2 - 1) = 0', d0['int1'], 1e-40)
    ck.small('averaged equation int dVeff/B^2 = 0', d0['int2'], 1e-40)
    ck.small('region-II Komar relation', Tp * Sp - Tm * Sm - d0['komar'], 1e-40)


def mixed_bc_check(w=mp.mpf('0.3'), Q1=mp.mpf(1), Q2=mp.mpf(2)):
    """Prop. 4.1(3): on the slice Sigma = W'(phi_inf) with W = w phi^2/2, the first law is
    d(M + W) = T+ dS+ (fixed charges).  The slice is M(phi) = (e^{-2phi}Q2^2 - e^{2phi}Q1^2)/(2 W'(phi))."""
    W = lambda p: w * p ** 2 / 2
    Wp = lambda p: w * p
    Mof = lambda p: (mp.e ** (-2 * p) * Q2 ** 2 - mp.e ** (2 * p) * Q1 ** 2) / (2 * Wp(p))
    def H(p):
        return Mof(p) + W(p)
    for p0 in (mp.mpf('0.1'), mp.mpf('0.2')):
        h = mp.mpf('1e-15')
        dH = (H(p0 + h) - H(p0 - h)) / (2 * h)
        Sp = lambda p: data(Mof(p), p, Q1, Q2)['+']['S']
        dS = (Sp(p0 + h) - Sp(p0 - h)) / (2 * h)
        d0 = data(Mof(p0), p0, Q1, Q2)
        print('   mixed b.c. W = %.2f phi^2/2 at phi_inf = %s (M = %s, Delta = %s): d(M+W) - T+ dS+ = %s ;  Sigma - W\' = %s' % (
            w, p0, mp.nstr(Mof(p0), 6), mp.nstr(d0['Delta'], 6), mp.nstr(dH - d0['+']['T'] * dS, 3), mp.nstr(d0['Sigma'] - Wp(p0), 3)))
        ck.small('mixed b.c. at phi_inf = %s: d(M+W) = T+ dS+' % p0, dH - d0['+']['T'] * dS, 1e-20)
        ck.small('mixed b.c. at phi_inf = %s: on the slice Sigma = W\'' % p0, d0['Sigma'] - Wp(p0), 1e-40)


if __name__ == '__main__':
    check(mp.mpf(3), mp.mpf('0.2'), mp.mpf(1), mp.mpf(2))
    check(mp.mpf('2.5'), mp.mpf('-0.7'), mp.mpf('0.6'), mp.mpf('1.3'))
    print('--- Prop. 4.1(3): mixed boundary condition Sigma = dW/dphi_inf, first law d(M+W) = T+ dS+ at fixed charges')
    mixed_bc_check()
    ck.finish()
