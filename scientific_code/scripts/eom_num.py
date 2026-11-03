r"""
eom_num.py -- numerical residuals of the 4D field equations for
    L = (1/16pi)[R - 2 g_ij dphi^i.dphi^j - f(phi) F.F - 2V] eps + (1/16pi) 2 ftilde(phi) F^F
on configurations depending on (r, th) in coordinates (t, r, th, ph).
Equations used:
    R_mn = 2 g_ij d_m phi^i d_n phi^j + 2 f (F_ma F_n^a - 1/4 g_mn F^2) + V g_mn
    nabla_m ( f F^{mn} + ftilde (*F)^{mn} ) = 0
    4 nabla_m(g_ij nabla^m phi^j) - 2 d_i g_jk dphi^j.dphi^k - d_i f F^2 - d_i ftilde F.(*F) - 2 d_i V = 0
      (F.(*F) = F_mn (*F)^{mn};  the theta-term in the Lagrangian density is  -ftilde F.(*F))
Second derivatives are obtained from symbolic differentiation of the component expressions.
"""
import itertools
import sympy as sp
import mpmath as mp

t, r, th, ph = sp.symbols('t r theta phi', real=True)
X = [t, r, th, ph]
VARS = (r, th)


def _derivs(expr, args):
    """value, first and second derivatives w.r.t. (r, th) as mpmath callables."""
    out = {(): sp.lambdify(args, expr, 'mpmath')}
    for i, v in enumerate(VARS):
        d1 = sp.diff(expr, v)
        out[(i + 1,)] = sp.lambdify(args, d1, 'mpmath')
        for j, w in enumerate(VARS):
            if j >= i:
                out[(i + 1, j + 1)] = sp.lambdify(args, sp.diff(d1, w), 'mpmath')
    return out


def make_residuals(g, Avec, scalars, gsig, fcoup, ftcoup, Vpot, params):
    args = (r, th) + tuple(params)
    D = 4
    gD = [[_derivs(g[i, j], args) for j in range(D)] for i in range(D)]
    AD = [_derivs(Avec[i], args) for i in range(D)]
    n = len(scalars)
    sD = [_derivs(scalars[i], args) for i in range(n)]
    ss = sp.symbols('s0:%d' % max(n, 1))[:n]
    gsC = sp.lambdify(ss, gsig(list(ss)), 'mpmath')
    dgsC = [sp.lambdify(ss, sp.diff(gsig(list(ss)), ss[k]), 'mpmath') for k in range(n)]
    fC = sp.lambdify(ss, fcoup(list(ss)), 'mpmath')
    ftC = sp.lambdify(ss, ftcoup(list(ss)), 'mpmath')
    VC = sp.lambdify(ss, Vpot(list(ss)), 'mpmath')
    dfC = [sp.lambdify(ss, sp.diff(fcoup(list(ss)), ss[k]), 'mpmath') for k in range(n)]
    dftC = [sp.lambdify(ss, sp.diff(ftcoup(list(ss)), ss[k]), 'mpmath') for k in range(n)]
    dVC = [sp.lambdify(ss, sp.diff(Vpot(list(ss)), ss[k]), 'mpmath') for k in range(n)]

    def res(rv, tv, pv):
        A_ = (rv, tv) + tuple(pv)
        def d(Dct, idx):   # idx: tuple of coordinate indices in 0..3 ; only r(1), th(2) nonzero
            if any(i in (0, 3) for i in idx):
                return mp.mpf(0)
            key = tuple(sorted(idx))
            return Dct[key](*A_)
        gm = mp.matrix([[d(gD[i][j], ()) for j in range(D)] for i in range(D)])
        gi = gm ** -1
        dgm = [mp.matrix([[d(gD[i][j], (a,)) for j in range(D)] for i in range(D)]) for a in range(D)]
        ddg = [[mp.matrix([[d(gD[i][j], (a, b)) for j in range(D)] for i in range(D)]) for b in range(D)] for a in range(D)]
        dgi = [-gi * dgm[a] * gi for a in range(D)]
        # Gamma^l_{ab} and its derivatives
        def Gam_low(a, b, c, dg):     # Gamma_{a b c} = 1/2 (d_b g_ac + d_c g_ab - d_a g_bc)
            return (dg[b][a, c] + dg[c][a, b] - dg[a][b, c]) / 2
        Gl = [[[Gam_low(a, b, c, dgm) for c in range(D)] for b in range(D)] for a in range(D)]
        Gam = [[[sum(gi[l, a] * Gl[a][b][c] for a in range(D)) for c in range(D)] for b in range(D)] for l in range(D)]
        dGam = [[[[None] * D for _ in range(D)] for _ in range(D)] for _ in range(D)]   # d_e Gamma^l_{bc}
        for e in range(D):
            dg_e = [ddg[a][e] for a in range(D)]   # d_e (d_a g)
            for l in range(D):
                for b in range(D):
                    for c in range(D):
                        s = 0
                        for a in range(D):
                            dGl = (dg_e[b][a, c] + dg_e[c][a, b] - dg_e[a][b, c]) / 2
                            s += dgi[e][l, a] * Gl[a][b][c] + gi[l, a] * dGl
                        dGam[e][l][b][c] = s
        Ric = mp.matrix(D, D)
        for b in range(D):
            for c in range(D):
                s = 0
                for a in range(D):
                    s += dGam[a][a][b][c] - dGam[c][a][b][a]
                    for e in range(D):
                        s += Gam[a][a][e] * Gam[e][b][c] - Gam[a][c][e] * Gam[e][b][a]
                Ric[b, c] = s
        # matter
        sv = [d(sD[i], ()) for i in range(n)]
        ds = [[d(sD[i], (a,)) for a in range(D)] for i in range(n)]
        dds = [[[d(sD[i], (a, b)) for b in range(D)] for a in range(D)] for i in range(n)]
        Gs = gsC(*sv) if n else None
        dA = [[d(AD[i], (a,)) for a in range(D)] for i in range(D)]
        ddA = [[[d(AD[i], (a, b)) for b in range(D)] for a in range(D)] for i in range(D)]
        F = mp.matrix(D, D)
        dF = [mp.matrix(D, D) for _ in range(D)]
        for m in range(D):
            for nn in range(D):
                F[m, nn] = dA[nn][m] - dA[m][nn]
                for e in range(D):
                    dF[e][m, nn] = ddA[nn][m][e] - ddA[m][nn][e]
        Fup = gi * F * gi
        FF = sum(F[m, nn] * Fup[m, nn] for m in range(D) for nn in range(D))
        sg = mp.sqrt(-mp.det(gm))
        def star_up(Fl, s_):
            # (*F)^{mn} = 1/2 eps^{mnab} F_ab, eps^{0123} = -1/s_
            S = mp.matrix(D, D)
            for p in itertools.permutations(range(4)):
                sign = sp.combinatorics.Permutation(list(p)).signature()
                m, nn, a, b = p
                S[m, nn] += sign * (-1 / s_) * Fl[a, b] / 2
            return S
        sFup = star_up(F, sg)
        FsF = sum(F[m, nn] * sFup[m, nn] for m in range(D) for nn in range(D))
        fv = fC(*sv); ftv = ftC(*sv); Vv = VC(*sv)
        Ein = mp.matrix(D, D)
        for m in range(D):
            for nn in range(D):
                T = 0
                for i in range(n):
                    for j in range(n):
                        T += 2 * Gs[i, j] * ds[i][m] * ds[j][nn]
                FmFn = sum(F[m, a] * F[nn, b] * gi[a, b] for a in range(D) for b in range(D))
                T += 2 * fv * (FmFn - gm[m, nn] * FF / 4) + Vv * gm[m, nn]
                Ein[m, nn] = Ric[m, nn] - T
        # Maxwell: (1/sg) d_m ( sg (f F^{mn} + ft *F^{mn}) ) ; do it by numerical differentiation of the bracket
        def bracket(rv_, tv_, nu):
            A2 = (rv_, tv_) + tuple(pv)
            gm2 = mp.matrix([[gD[i][j][()](*A2) for j in range(D)] for i in range(D)])
            gi2 = gm2 ** -1
            sg2 = mp.sqrt(-mp.det(gm2))
            dA2 = [[mp.mpf(0), AD[i][(1,)](*A2), AD[i][(2,)](*A2), mp.mpf(0)] for i in range(D)]
            F2 = mp.matrix(D, D)
            for m in range(D):
                for nn in range(D):
                    F2[m, nn] = dA2[nn][m] - dA2[m][nn]
            sv2 = [sD[i][()](*A2) for i in range(n)]
            Fu2 = gi2 * F2 * gi2
            sF2 = star_up(F2, sg2)
            return [sg2 * (fC(*sv2) * Fu2[m, nu] + ftC(*sv2) * sF2[m, nu]) for m in range(D)]
        Max = []
        for nu in range(D):
            dr_ = mp.diff(lambda x: bracket(x, tv, nu)[1], rv)
            dth_ = mp.diff(lambda x: bracket(rv, x, nu)[2], tv)
            Max.append((dr_ + dth_) / sg)
        # scalars
        Scal = []
        for i in range(n):
            # 4 nabla_m (g_ij nabla^m phi^j) = 4/sg d_m (sg g_ij g^{mn} d_n phi^j)
            def flux(rv_, tv_, m):
                A2 = (rv_, tv_) + tuple(pv)
                gm2 = mp.matrix([[gD[a][b][()](*A2) for b in range(D)] for a in range(D)])
                gi2 = gm2 ** -1
                sg2 = mp.sqrt(-mp.det(gm2))
                sv2 = [sD[k][()](*A2) for k in range(n)]
                G2 = gsC(*sv2)
                dsv2 = [[mp.mpf(0), sD[k][(1,)](*A2), sD[k][(2,)](*A2), mp.mpf(0)] for k in range(n)]
                return sg2 * sum(G2[i, j] * gi2[m, nn] * dsv2[j][nn] for j in range(n) for nn in range(D))
            box = (mp.diff(lambda x: flux(x, tv, 1), rv) + mp.diff(lambda x: flux(rv, x, 2), tv)) / sg
            dG = dgsC[i](*sv)
            kin = sum(dG[j, k] * ds[j][m] * ds[k][nn] * gi[m, nn] for j in range(n) for k in range(n) for m in range(D) for nn in range(D))
            Scal.append(4 * box - 2 * kin - dfC[i](*sv) * FF - dftC[i](*sv) * FsF - 2 * dVC[i](*sv))
        return Ein, Max, Scal
    return res
