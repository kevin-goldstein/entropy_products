r"""
static_K.py -- reduce the Iyer-Wald (D-2)-form k_xi = delta Q_xi - xi.Theta, xi = d/dt,
on the static ansatz  ds^2 = -A dt^2 + dr^2/A + B^2 dOmega_{D-2}^2  (A = a^2, B = b),
for an arbitrary variation (dA, dB) of the metric functions.

Output: K_grav(r) = \oint_{S_r} k_xi (gravity part), as a function of A, B, dA, dB and derivatives.
"""
import sympy as sp
from cps import k_grav_tr, sphere_metric, sphere_volume

r, t = sp.symbols('r t')
A = sp.Function('A')(r)
B = sp.Function('B')(r)
dA = sp.Function('dA')(r)
dB = sp.Function('dB')(r)
eps = sp.Symbol('epsilon_var')


def K_grav(D, verbose=False):
    angles = sp.symbols('th1:%d' % (D - 1))
    x = [t, r] + list(angles)
    sph = sphere_metric(D - 2, angles)
    g = sp.diag(-A, 1 / A, *[B ** 2 * s for s in sph])
    h = sp.diag(-dA, -dA / A ** 2, *[2 * B * dB * s for s in sph])
    xi = [1] + [0] * (D - 1)
    angfac = sp.prod([sp.sin(angles[i]) ** (D - 3 - i) for i in range(D - 2)])

    def sqrtg(ge, e):
        # first-order exact for the diagonal static ansatz
        return sp.sqrt((A + e * dA) * (1 / A - e * dA / A ** 2)) * (B + e * dB) ** (D - 2) * angfac

    k16 = k_grav_tr(g, h, x, xi, eps=eps, sqrtg=sqrtg)   # 16 pi sqrt(-g) k^{tr}
    kred = sp.simplify(k16 / angfac)
    K = sp.simplify(sphere_volume(D - 2) / (16 * sp.pi) * kred)
    return sp.expand(K)


if __name__ == '__main__':
    import checklib as ck
    for D in (4, 5, 6):
        K = K_grav(D)
        print('D =', D, ':  K_grav =', sp.factor_terms(K))
        expected = (D - 2) * sphere_volume(D - 2) / (16 * sp.pi) * B ** (D - 3) * (
            sp.diff(A, r) * dB - sp.diff(B, r) * dA - 2 * A * sp.diff(dB, r))
        ck.zero('D=%d: K_grav = (D-2) Omega B^(D-3) [A\' dB - B\' dA - 2 A dB\'] / 16 pi' % D, K - expected)
    ck.finish()
