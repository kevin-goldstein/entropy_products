"""Independent formal recursion using only exact standard-library fractions.

Coefficients lie in Q[s]/(s^2-S); polynomials are in z. No author code imported.
Only sources through the first nonzero obstruction are solved: a nonzero
projection is reported, not replaced by a regular polynomial.
"""
from fractions import Fraction as F
from math import factorial
import json
import os
from pathlib import Path

S=F(0)

def c(x=0): return (F(x),F(0))
def ca(a,b): return (a[0]+b[0],a[1]+b[1])
def cn(a): return (-a[0],-a[1])
def cm(a,b): return (a[0]*b[0]+S*a[1]*b[1],a[0]*b[1]+a[1]*b[0])
def cs(a,f): return (a[0]*f,a[1]*f)
ZERO=c()

def trim(a):
    while len(a)>1 and a[-1]==ZERO:a.pop()
    return a
def const(x):return [x]
def add(a,b):
    out=[ZERO]*max(len(a),len(b))
    for k,x in enumerate(a):out[k]=ca(out[k],x)
    for k,x in enumerate(b):out[k]=ca(out[k],x)
    return trim(out)
def scale(a,k):return trim([cs(x,F(k)) for x in a])
def mul(a,b):
    out=[ZERO]*(len(a)+len(b)-1)
    for i,x in enumerate(a):
        if x==ZERO:continue
        for j,y in enumerate(b):
            if y!=ZERO:out[i+j]=ca(out[i+j],cm(x,y))
    return trim(out)
def deriv(a):return trim([cs(a[k],k) for k in range(1,len(a))] or [ZERO])
def integ(a):return [ZERO]+[cs(v,F(1,k+1)) for k,v in enumerate(a)]
def endpoint(a,z):
    ans=ZERO
    for k,x in enumerate(a):ans=ca(ans,cs(x,z**k))
    return ans
def area(a):
    ans=ZERO
    for k in range(0,len(a),2):ans=ca(ans,cs(a[k],F(2,k+1)))
    return ans
def impose_metric(source,slope):
    first=integ(source)
    first[0]=ca(slope,cn(endpoint(first,1)))
    second=integ(first)
    second[0]=cn(endpoint(second,1))
    return trim(second)
def legendre(n):
    prev,cur=[c(1)],[ZERO,c(1)]
    if n==0:return prev
    for k in range(1,n):
        prev,cur=cur,scale(add(scale([ZERO]+cur,2*k+1),scale(prev,-k)),F(1,k+1))
    return cur
def solve_regular(src,nu):
    d=max(len(src)-1,nu)
    a=[ZERO]*(d+3);right=src+[ZERO]*(d+3-len(src))
    for k in range(d,-1,-1):
        v=ca(right[k],cs(a[k+2],(k+2)*(k+1)))
        denom=k*(k+1)-nu*(nu+1)
        if denom==0:
            assert v==ZERO,('resonance',nu,v)
        else:a[k]=cs(v,F(1,denom))
    a=trim(a)
    return add(a,mul(const(cn(endpoint(a,1))),legendre(nu)))
def series_mul(a,b,N):
    out={}
    for i,x in a.items():
        for j,y in b.items():
            if i+j<=N:out[i+j]=add(out.get(i+j,[ZERO]),mul(x,y))
    return out

def run(nu,Svalue):
    global S
    S=F(Svalue)
    p=-2*nu*(nu+1)
    v=[c(1),ZERO,c(-p)]
    for j in range(3,7):v.append(ca(cm((F(0),F(1)),v[j-1]),cs(v[j-2],-p)))
    phi={1:legendre(nu)};h={0:[c(1)]}
    out={}
    for n in range(2,6):
        dp={k:deriv(x) for k,x in phi.items()}
        metric=series_mul(h,series_mul(dp,dp,n),n).get(n,[ZERO])
        h[n]=impose_metric(scale(metric,-1),cs(v[n],-F(1,2*factorial(n))))
        hh=series_mul(h,h,n)
        inv={0:[c(1)]}
        for k in range(1,n+1):
            inv[k]=[ZERO]
            for j in range(1,k+1):
                inv[k]=add(inv[k],scale(mul(hh.get(j,[ZERO]),inv[k-j]),-1))
        power={0:[c(1)]};force={}
        for j in range(1,n+1):
            power=series_mul(power,phi,n)
            for k,val in power.items():
                force[k]=add(force.get(k,[ZERO]),mul(val,const(cs(v[j+1],F(1,2*factorial(j))))))
        src=series_mul(force,inv,n).get(n,[ZERO])
        projection=cs(area(mul(legendre(nu),src)),F((-1)**nu,2))
        out[str(n)]={'rational':str(projection[0]),'s_coefficient':str(projection[1]),
                     'decimal':float(projection[0])+float(projection[1])*float(S)**.5}
        if projection!=ZERO:
            break
        phi[n]=solve_regular(src,nu)
        # The differential polynomial and outer endpoint are direct independent checks.
        op=add(deriv(mul([c(-1),ZERO,c(1)],deriv(phi[n]))),scale(phi[n],-nu*(nu+1)))
        assert add(op,scale(src,-1))==[ZERO]
        assert endpoint(phi[n],1)==ZERO
    return {'nu':nu,'s_squared':str(S),'coefficients':out,
            'regular_scalar_endpoint_coefficients':{str(k):[str(q) for q in endpoint(a,-1)] for k,a in phi.items()},
            'metric_endpoint_coefficients':{str(k):[str(q) for q in endpoint(a,-1)] for k,a in h.items()}}

if __name__=='__main__':
    results=[run(7,F(5735000,109999)),run(9,F(4034475760,41535849))]
    # These known-value assertions are regressions; the independent evidence is
    # the exact source projection and differential-equation residuals above.
    assert results[0]['coefficients']['5']['rational']=='-50477428324786176/709563393571462669'
    assert results[1]['coefficients']['5']['rational']=='-13601636237055868467414220800/31558405059126012246505659137'
    assert all(all(r['coefficients'][str(k)]['rational']=='0' and r['coefficients'][str(k)]['s_coefficient']=='0' for k in (2,3,4)) for r in results)
    print(json.dumps(results,indent=2))
    output=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT', Path(__file__).resolve().parents[1]/'results/data'))
    folder=output/'controls';folder.mkdir(parents=True,exist_ok=True)
    (folder/'exact_obstruction_control.json').write_text(json.dumps(results,indent=2)+'\n')
