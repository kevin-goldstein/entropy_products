from fractions import Fraction as F
import math

def add(p,q):
    r=[F(0)]*max(len(p),len(q))
    for k,v in enumerate(p):r[k]+=v
    for k,v in enumerate(q):r[k]+=v
    return r
def scale(p,c):return [x*c for x in p]
def mul(p,q):
    r=[F(0)]*(len(p)+len(q)-1)
    for i,x in enumerate(p):
        for j,y in enumerate(q):r[i+j]+=x*y
    return r
def deriv(p):return [p[k]*k for k in range(1,len(p))]
def integ(p):return [F(0)]+[x/F(k+1) for k,x in enumerate(p)]
def integral(p):return sum(2*x/F(k+1) for k,x in enumerate(p) if k%2==0)
def value1(p):return sum(p)
def legendre(n):
    p0,p1=[F(1)],[F(0),F(1)]
    if n==0:return p0
    for k in range(1,n):p0,p1=p1,scale(add(scale([F(0)]+p1,2*k+1),scale(p0,-k)),F(1,k+1))
    return p1
def solveL_even(rhs,lam):
    p=[F(0)]*len(rhs)
    for k in range(len(p)-1,-1,-1):
        val=rhs[k]
        if k+2<len(p):val+=(k+2)*(k+1)*p[k+2]
        if k*(k+1)==lam:
            assert val==0
        else:p[k]=val/F(k*(k+1)-lam)
    return p
def obstruction(n,s):
    lam=n*(n+1);p=legendre(n)
    pp=mul(p,p)
    beta=integ(integ(scale(mul(deriv(p),deriv(p)),-1)))
    beta[1]=-F(lam,2)-value1(deriv(beta))
    beta[0]=-value1(beta)
    p2=solveL_even(scale(pp,F(lam*s,2)),lam)
    p2=add(p2,scale(p,-value1(p2)))
    source=add(add(scale(mul(p,p2),lam*s),scale(mul(pp,p),F(lam*(s*s+2*lam),6))),scale(mul(beta,p),-2*lam))
    return F((-1)**n,2)*integral(mul(p,source))

def value_minus1(p):
    return sum(x*(-1)**k for k,x in enumerate(p))

def trim(p):
    while len(p)>1 and p[-1]==0:p.pop()
    return p

def n3_quartic():
    """Exact coupled expansion, with epsilon equal to the outer scalar value.

    The projective radius B=b/b_RN obeys Bzz=-B phi_z**2. The scalar
    equation is ((z*z-1)phi_z)_z=alpha*sinh(alpha*phi)/(2*B*B).
    Its horizon constraint gives B_z(1)=-(cosh(alpha*epsilon)-1)/2.
    """
    lam=12
    p=legendre(3)
    beta=integ(integ(scale(mul(deriv(p),deriv(p)),-1)))
    beta[1]=-F(lam,2)-value1(deriv(beta))
    beta[0]=-value1(beta)
    source0=add(scale(mul(mul(p,p),p),F(lam*lam,3)),
                scale(mul(beta,p),-2*lam))
    lam2=-integral(mul(p,source0))/integral(mul(p,p))
    source=add(source0,scale(p,lam2))
    q=solveL_even(source,lam)
    q=add(q,scale(p,-value1(q)))
    gamma=integ(integ(add(scale(mul(beta,mul(deriv(p),deriv(p))),-1),
                           scale(mul(deriv(p),deriv(q)),-2))))
    gamma[1]=-F(1,2)*(lam2+F(lam*lam,6))-value1(deriv(gamma))
    gamma[0]=-value1(gamma)
    assert value1(beta)==value_minus1(beta)==0
    assert value1(q)==value_minus1(q)==0
    assert lam2==F(-96,143)
    assert value_minus1(gamma)==F(-48,1001)
    return dict(beta=trim(beta),scalar_cubic=trim(q),gamma=trim(gamma),
                lambda_quadratic=lam2,nu_quadratic=lam2/7,
                gamma_inner=value_minus1(gamma),
                deficit_quartic=-2*value_minus1(gamma))

if __name__=='__main__':
    import json
    from pathlib import Path
    import os
    result=n3_quartic()
    result={k:([str(x) for x in v] if isinstance(v,list) else str(v))
            for k,v in result.items()}
    text=json.dumps(result,indent=2)+'\n'
    print(text,end='')
    out=Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1]/"results/data"))/"branch"
    out.mkdir(parents=True, exist_ok=True)
    (out/"quartic_exact.json").write_text(text)
