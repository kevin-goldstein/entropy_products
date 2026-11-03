"""Independent polynomial/residue checks for the complex-root expansion.

The analytic derivation is in the paper appendix on complex continuation. These finite-precision checks
exercise the complex roots, reciprocal-temperature moments and implicit root
variations; they make no claim about global analyticity of hairy solutions.
"""
from pathlib import Path
import os
import json
import numpy as np

def test_rnads(M,W,ell):
    co=np.array([1/ell**2,0.,1.,-2*M,W])
    rr=np.roots(co); pp=np.polyval(np.polyder(co),rr)
    ss=np.pi*rr**2;tt=pp/(4*np.pi*rr**2)
    moments=np.array([np.sum(rr**k/pp) for k in range(4)])
    drM=2*rr/pp;drW=-1/pp;drl=2*rr**4/(ell**3*pp)
    relative=lambda got,exact:float(abs(got-exact)/max(1,abs(exact)))
    checks=dict(moment_zero=max(float(abs(moments[k])) for k in range(3)),
      moment_three=relative(moments[3],ell**2),
      inverse_T=abs(np.sum(1/tt)),inverse_TS=abs(np.sum(1/(tt*ss))),
      product=relative(np.prod(ss),np.pi**4*ell**4*W**2),
      sum=relative(np.sum(ss),-2*np.pi*ell**2),
      origin_residue=relative(np.sum(1/(rr*pp)),-1/W),
      firstlaw_M=max(abs(tt*(2*np.pi*rr*drM)-1)),
      firstlaw_W=max(abs(tt*(2*np.pi*rr*drW)+1/(2*rr))),
      firstlaw_l=max(abs(tt*(2*np.pi*rr*drl)-rr**3/ell**3)),
      dlogprod_M=abs(np.sum(2*drM/rr)),
      dlogprod_W=relative(np.sum(2*drW/rr),2/W),
      dlogprod_l=relative(np.sum(2*drl/rr),4/ell))
    checks={k:float(v) for k,v in checks.items()}
    real=rr[(abs(rr.imag)<1e-10)&(rr.real>0)].real
    return dict(M=M,W=W,l=ell,max_error=max(checks.values()),checks=checks,
                physical_radii=sorted(real.tolist()),
                physical_entropy_product=float(np.prod(np.pi*real**2)) if len(real)==2 else None)

def test_general():
    rng=np.random.default_rng(741)
    out=[]
    for n in range(2,8):
        roots=rng.normal(size=n)+1j*rng.normal(size=n)
        lead=1.3+.4j
        co=lead*np.poly(roots)
        pp=np.polyval(np.polyder(co),roots)
        actual=np.array([sum(roots**k/pp) for k in range(n)])
        exact=np.r_[np.zeros(n-1,dtype=complex),1/lead]
        out.append(dict(degree=n,max_error=float(max(abs(actual-exact)))))
    return out

if __name__=='__main__':
    result=dict(RNAdS=[test_rnads(M,16.,1.) for M in [9.,10.,11.]],
                general_polynomials=test_general())
    assert max(x['max_error'] for x in result['RNAdS'])<1e-10
    assert max(x['max_error'] for x in result['general_polynomials'])<1e-10
    text=json.dumps(result,indent=2)+'\n'
    out=Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1]/"results/data"))/"controls"
    out.mkdir(parents=True,exist_ok=True)
    (out/"complex_validation.json").write_text(text)
    print(text)
