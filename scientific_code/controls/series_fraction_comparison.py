"""Portable exact tuned-branch recurrence and comparison to unrounded shooting data.

The recurrence is adapted from the supplied tuned_branch_series.py, with Python
Fraction polynomials replacing python-flint.  This is a backend reproduction,
not an independent high-order derivation.  See coupling_series_control.py for
an independent SymPy derivation through the first new coefficients.

Usage: python series_fraction_comparison.py --table path/to/amplitude_scan.csv
       --output path/to/series_comparison.csv [--orders 29 35]
An odd order N computes coupling and entropy series through epsilon**(N-1).
The manuscript comparison uses power 34; power 28 is retained as a truncation
diagnostic. The tolerances are empirical agreement checks, not accuracy bounds.
Only the Python standard library and the local fraction_polynomial.py are used.
"""
import argparse
import csv
import hashlib
import os
from decimal import Decimal, localcontext
from fractions import Fraction
from math import factorial
from pathlib import Path
from fraction_polynomial import FractionPolynomial

fmpq=Fraction
fmpq_poly=FractionPolynomial
z=fmpq_poly([0,1])

def solve_L_poly(nu,source):
    degree=max(source.degree(),nu)
    coefficients=[Fraction(0)]*(degree+3)
    for k in range(degree,-1,-1):
        numerator=source[k]+(k+2)*(k+1)*coefficients[k+2]
        denominator=k*(k+1)-nu*(nu+1)
        if denominator:
            coefficients[k]=numerator/denominator
        else:
            assert numerator==0, (nu,k,numerator)
    polynomial=fmpq_poly(coefficients)
    polynomial-=polynomial(1)*fmpq_poly.legendre_p(nu)
    assert ((z*z-1)*polynomial.derivative()).derivative()-nu*(nu+1)*polynomial==source
    return Fraction(0),polynomial,Fraction(0)

def ser_mul(a, b, N):
    out = [fmpq(0)] * (N + 1)
    for i, x in enumerate(a[:N + 1]):
        if x != 0:
            for j, y in enumerate(b[:N + 1 - i]):
                out[i + j] += x * y
    return out

def pmul(a, b, n):
    """series (dict order -> fmpq_poly) product truncated at order n"""
    out = {}
    for i, x in a.items():
        for j, y in b.items():
            if i + j <= n:
                out[i + j] = out.get(i + j, fmpq_poly([])) + x * y
    return out

def lindstedt(nu0, N, tune=True, bc_sign=-1):
    """symmetric model V/V_* = cosh(alpha delta), alpha^2 = 2 lambda^2, lambda^2 = nu0(nu0+1) - sum m_k eps^k.
    Returns m, phi (dict n -> poly), h (dict n -> poly), Vp ([V_+/V_*]_n).  With tune=False (m = 0) it stops at the
    first order whose source has a P_nu0 component and returns that order and the projection."""
    lam0 = nu0 * (nu0 + 1)
    P = fmpq_poly.legendre_p(nu0)
    norm = fmpq(2, 2 * nu0 + 1)                       # int P^2
    m = [fmpq(0)] * (N + 1)
    phi, h, Vp = {1: P}, {0: fmpq_poly([1])}, [fmpq(1)] + [fmpq(0)] * N

    def couplings():
        lam2 = [fmpq(lam0)] + [-m[k] for k in range(1, N + 1)]
        two = [2 * x for x in lam2]
        powk = [[fmpq(1)] + [fmpq(0)] * N]            # (2 lambda^2)^k
        for k in range(1, N // 2 + 2):
            powk.append(ser_mul(powk[-1], two, N))
        return powk

    for n in range(2, N + 1):
        powk = couplings()                            # uses m_k, k <= n - 2
        # [cosh(alpha eps)]_n = sum_k [(2 lambda^2)^k]_{n - 2k} / (2k)!
        Vp[n] = sum((powk[k][n - 2 * k] / factorial(2 * k) for k in range(1, n // 2 + 1)), fmpq(0))
        dphi = {k: x.derivative() for k, x in phi.items()}
        acc = fmpq_poly([])
        for a_, ha in h.items():
            for b_, pb in dphi.items():
                if (n - a_ - b_) in dphi:
                    acc += ha * pb * dphi[n - a_ - b_]
        hp = (-acc).integral()
        hp = hp - hp(1) + bc_sign * Vp[n] / 2         # h_n'(1) = -[V_+]_n/2
        hn = hp.integral()
        h[n] = hn - hn(1)
        H2 = pmul(h, h, n)
        g = {0: fmpq_poly([1])}
        for mm in range(1, n):
            a2 = fmpq_poly([])
            for k in range(1, mm + 1):
                if k in H2 and (mm - k) in g:
                    a2 += H2[k] * g[mm - k]
            g[mm] = -a2
        # V'/(2 V_*) = sum_{k>=0} (2 lambda^2)^{k+1} delta^{2k+1} / (2 (2k+1)!)
        F, D, D2 = {}, dict(phi), pmul(phi, phi, n)
        for k in range(0, n // 2 + 1):
            coef = [x / (2 * factorial(2 * k + 1)) for x in powk[k + 1]]
            for i, x in D.items():
                for q in range(0, n + 1 - i):
                    if coef[q] != 0:
                        F[i + q] = F.get(i + q, fmpq_poly([])) + coef[q] * x
            D = pmul(D, D2, n)
        sn = fmpq_poly([])
        for k, x in F.items():
            if (n - k) in g:
                sn += x * g[n - k]
        I = (P * sn).integral()
        proj = I(1) - I(-1)
        if not tune:
            if proj != 0:
                return dict(resonant_order=n, projection=proj, phi=phi, h=h)
            m[n - 1] = fmpq(0)
        else:
            m[n - 1] = proj / norm                    # lambda^2 contains -m_{n-1} eps^{n-1}: adds -m_{n-1} P here
        sig = sn - m[n - 1] * P
        kap, p, kg = solve_L_poly(nu0, sig)
        assert kap == 0 and kg == 0, 'source still resonant'
        phi[n] = p
    return dict(m=m, phi=phi, h=h, Vp=Vp, N=N)

def nu_series(nu0, m, N):
    """nu = nu0 - d(eps): (nu0 - d)(nu0 + 1 - d) = nu0(nu0+1) - m  <=>  d = (m + d^2)/(2 nu0 + 1)"""
    d = [fmpq(0)] * (N + 1)
    for _ in range(N + 1):
        d = [(x + y) / (2 * nu0 + 1) for x, y in zip(m, ser_mul(d, d, N))]
    return d

def at_minus1(res):
    N = res['N']
    return [res['h'][k](-1) if k in res['h'] else fmpq(0) for k in range(N + 1)]

def product_series(res):
    hm = at_minus1(res)
    return ser_mul(hm, hm, res['N'])

def coupling_identity(nu0, res, sign=1):
    """LHS - RHS of  d ln h(-1)/d eps = (sign/2) (d lambda^2/d eps) int_{-1}^{1} [d_{lambda^2} cosh(alpha delta)]/h^2 dz,
    d_{lambda^2} cosh(alpha delta) = delta sinh(alpha delta)/alpha = sum_k (2 lambda^2)^k delta^{2k+2}/(2k+1)!"""
    N, m, phi, h = res['N'], res['m'], res['phi'], res['h']
    lam2 = [fmpq(nu0 * (nu0 + 1))] + [-m[k] for k in range(1, N + 1)]
    H = at_minus1(res)
    Hp = [(k + 1) * H[k + 1] for k in range(N)] + [fmpq(0)]
    inv = [fmpq(1)] + [fmpq(0)] * N
    for k in range(1, N + 1):
        inv[k] = -sum((H[j] * inv[k - j] for j in range(1, k + 1)), fmpq(0))
    lhs = ser_mul(Hp, inv, N - 1)
    dl = [(k + 1) * lam2[k + 1] for k in range(N)] + [fmpq(0)]
    H2 = pmul(h, h, N)
    g = {0: fmpq_poly([1])}
    for mm in range(1, N + 1):
        a2 = fmpq_poly([])
        for k in range(1, mm + 1):
            if k in H2 and (mm - k) in g:
                a2 += H2[k] * g[mm - k]
        g[mm] = -a2
    d2 = pmul(phi, phi, N)
    Dk, tl, W = dict(d2), [fmpq(1)] + [fmpq(0)] * N, {}
    for k in range(0, N // 2 + 1):
        coef = [x / factorial(2 * k + 1) for x in tl]
        for i, x in Dk.items():
            for q in range(N + 1 - i):
                if coef[q] != 0:
                    W[i + q] = W.get(i + q, fmpq_poly([])) + coef[q] * x
        Dk = pmul(Dk, d2, N)
        tl = ser_mul(tl, [2 * x for x in lam2], N)
    I = [fmpq(0)] * (N + 1)
    for k, x in pmul(W, g, N).items():
        X = x.integral()
        I[k] = X(1) - X(-1)
    rhs = [sign * x / 2 for x in ser_mul(dl, I, N - 1)]
    return [a - b for a, b in zip(lhs[:N - 1], rhs[:N - 1])]

def dec(value):
    return Decimal(value.numerator)/Decimal(value.denominator)

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[1]
    output_root=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT',root/'results/data'))
    generated_table=output_root/'branch/amplitude_scan.csv'
    default_table=generated_table if generated_table.is_file() else root/'data/generated/branch/amplitude_scan.csv'
    parser.add_argument('--table',type=Path,default=default_table)
    parser.add_argument('--output',type=Path,default=output_root/'controls/series_comparison.csv')
    parser.add_argument('--orders',type=int,nargs='+',default=[29,35])
    args=parser.parse_args()
    print('Input table:',args.table.resolve(),flush=True)
    print('Input SHA-256:',hashlib.sha256(args.table.read_bytes()).hexdigest(),flush=True)
    results={}
    for order in args.orders:
        assert order%2 and order>=7
        result=lindstedt(3,order)
        d=nu_series(3,result['m'],order)
        sigma=product_series(result)
        assert all(x==0 for x in coupling_identity(3,result))
        assert d[2]==Fraction(96,1001) and d[4]==-Fraction(38719078400,323969969323)
        assert sigma[4]==-Fraction(96,1001) and sigma[6]==Fraction(51807744,231175945)
        results[order]=(result,d,sigma)
        print('PASS: exact recurrence and coupling identity through epsilon^'+str(order-1),flush=True)
    with args.table.open(newline='') as stream:
        rows=list(csv.DictReader(stream))
    output=[]
    paper_amplitudes={Decimal(x) for x in ('0.025','0.05','0.1','0.15','0.2','0.3')}
    with localcontext() as context:
        context.prec=60
        for row in rows:
            epsilon=Decimal(row['epsilon'])
            for order,(result,d,sigma) in results.items():
                Lambda=12-sum(dec(c)*epsilon**k for k,c in enumerate(result['m']))
                alpha=(2*Lambda).sqrt()
                delta_nu=(7-(1+4*Lambda).sqrt())/2
                eta=sum(dec(c)*epsilon**k for k,c in enumerate(sigma))
                values=dict(epsilon=str(epsilon),series_order=order-1,
                    in_manuscript_table=epsilon in paper_amplitudes,
                    alpha_series=str(alpha),delta_nu_series=str(delta_nu),eta_series=str(eta),
                    alpha_shooting=row['alpha'],delta_nu_shooting=row['delta_nu'],eta_shooting=row['eta'],
                    alpha_difference=str(alpha-Decimal(row['alpha'])),
                    delta_nu_difference=str(delta_nu-Decimal(row['delta_nu'])),
                    eta_difference=str(eta-Decimal(row['eta'])))
                output.append(values)
                print('epsilon='+str(epsilon),'order='+str(order-1),
                      'alpha difference='+format(alpha-Decimal(row['alpha']),'.4E'),
                      'eta difference='+format(eta-Decimal(row['eta']),'.4E'),flush=True)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(output[0]))
        writer.writeheader();writer.writerows(output)
    # At epsilon=0.3 the power-28 truncation can partially cancel shooting error.
    # Use power 34 for the manuscript agreement check and retain both in the CSV.
    selected=[row for row in output if row['series_order']==34 and row['in_manuscript_table']]
    if selected:
        assert len(selected)==6 and {Decimal(row['epsilon']) for row in selected}==paper_amplitudes
        alpha_error=max(abs(Decimal(row['alpha_difference'])) for row in selected)
        eta_error=max(abs(Decimal(row['eta_difference'])) for row in selected)
        assert alpha_error<Decimal('3e-13') and eta_error<Decimal('1e-13')
        print('PASS: order34 versus the six manuscript Table2 rows:')
        print('  max |alpha difference| =',alpha_error)
        print('  max |eta difference| =',eta_error)
    print('Comparisons are empirical; no rigorous convergence radius or truncation bound is inferred.')

if __name__=='__main__':
    main()
