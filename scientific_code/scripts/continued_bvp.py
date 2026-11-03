r"""Guarded balanced continuation of the independently constructed tuned branch.

The fractional linear coordinate map is exact. The supplied BVP profile remains
numerical, including its regular Taylor endpoint approximations. Neither this
map nor convergence supplies a rigorous existence theorem for that profile.
No inward integration or fitted logarithmic coefficient is imported.
"""
from pathlib import Path
import os
import json
import numpy as np
from scipy.integrate import quad
from bvp_collocation import RM, RP, DELTA, BVPError, solve_branch
import checklib as ck


def balanced_sample(solution, radius):
    """Return (b,b',phi,phi') on the balanced representative of [RM,RP]."""
    bp, bm = float(solution.bp), float(solution.bm)
    if not np.isfinite([bp, bm]).all() or min(bp, bm) <= 0:
        raise BVPError('balancing requires two finite positive horizon radii')
    if not solution.collocation.success or solution.collocation.status != 0:
        raise BVPError('balancing requires successful complete collocation')
    r = np.asarray(radius, dtype=float)
    if r.ndim > 1 or not np.isfinite(r).all() or np.any(r < RM) or np.any(r > RP):
        raise BVPError('balanced sample lies outside the two horizons')
    k = bm / bp
    x = (r-RM)/DELTA
    d = 1+(k-1)*x
    mapped = RM+DELTA*k*x/d
    # Exact map is interval preserving; permit only its floating point roundoff.
    slack = 16*np.finfo(float).eps*max(1., abs(RP), abs(RM))
    if np.any(mapped < RM-slack) or np.any(mapped > RP+slack):
        raise BVPError('balancing coordinate map left the physical interval')
    mapped = np.clip(mapped, RM, RP)
    b, bd, phi, pd = solution.sample(mapped)
    result = np.asarray([d*b/np.sqrt(k), (k-1)*b/(DELTA*np.sqrt(k))+np.sqrt(k)*bd/d,
                         phi, k*pd/d**2])
    if not np.isfinite(result).all() or np.any(result[0] <= 0):
        raise BVPError('invalid balanced fields')
    return result


def potential_integrals(sample, alpha):
    """Independent quadrature, for validation only, on the whole closed strip."""
    result = []
    for sign in (1, -1):
        def integrand(r):
            b, _, phi, _ = sample(r)
            return np.exp(sign*alpha*phi)/b**2
        value, error = quad(integrand, RM, RP, epsabs=2e-11, epsrel=2e-11,
                            points=np.linspace(RM, RP, 17)[1:-1], limit=400)
        if not np.isfinite([value,error]).all() or error > 1e-9:
            raise BVPError('potential quadrature failed its error check')
        result.append(float(value))
    return result


def symbolic_map_checks():
    """Independent algebraic checks of the map before using numerical profiles."""
    import sympy as sp
    x,k,D,B,B1,B2,u1,W,Wphi=sp.symbols('x k D B B1 B2 u1 W Wphi',positive=True)
    d=1+(k-1)*x
    y=k*x/d
    jac=k/d**2
    P=D**2*x*(x-1)
    Pold=D**2*y*(y-1)
    Bnew=d*B/sp.sqrt(k)
    Bnew1=(k-1)*B/(D*sp.sqrt(k))+sp.sqrt(k)*B1/d
    Bnew2=sp.diff(Bnew1,x)/D+sp.diff(Bnew1,B)*B1*jac+sp.diff(Bnew1,B1)*B2*jac
    ck.zero('balancing map fixes inner endpoint',y.subs(x,0))
    ck.zero('balancing map fixes outer endpoint',y.subs(x,1)-1)
    ck.zero('balancing map radial Jacobian',sp.diff(y,x)-jac)
    ck.zero('balancing map preserves scalar flux',P*jac-Pold)
    ck.zero('balancing map preserves scalar equation with arbitrary potential derivative',
            jac*Wphi/(2*B**2)-Wphi/(2*Bnew**2))
    ck.zero('balancing map transforms b second derivative',Bnew2-k**sp.Rational(3,2)*B2/d**3)
    ck.zero('balancing map preserves radius equation',
            Bnew2.subs(B2,-B*u1**2)+Bnew*(jac*u1)**2)
    Hnew=-1+D*(2*x-1)*Bnew1/Bnew-P*(Bnew1/Bnew)**2+W/Bnew**2-P*(jac*u1)**2
    Hold=-1+D*(2*y-1)*B1/B-Pold*(B1/B)**2+W/B**2-Pold*u1**2
    ck.zero('balancing map preserves Hamiltonian constraint',Hnew-jac*Hold)
    ck.zero('balancing map preserves potential integration measure',jac/B**2-1/Bnew**2)
    ck.true('negative control: coordinate reparametrization without radius weight fails',
            sp.simplify(jac/B**2-1/B**2)!=0)


def run_checks():
    symbolic_map_checks()
    records=[]
    for order,cutoff,nodes in [(6,1e-4,401),(8,1e-4,601)]:
        solution=solve_branch(order=order,cutoff=cutoff,tolerance=1e-10,initial_nodes=nodes)
        d=solution.diagnostics()
        ck.small('source BVP independent ODE residual, N=%d'%order,d['off_grid_scaled_ode_max'],1e-8)
        ck.small('source BVP independent constraint, N=%d'%order,d['hamiltonian_constant_max_abs'],1e-9)
        R=solution.bp*solution.bm
        b,bd,phi,pd=balanced_sample(solution,np.linspace(RM,RP,1601))
        r=np.linspace(RM,RP,1601);P=(r-RP)*(r-RM);Pp=2*r-RP-RM
        H=-1+Pp*bd/b-P*bd**2/b**2+2*np.cosh(solution.alpha*phi)/b**2-P*pd**2
        ck.small_all('balanced endpoint area radii, N=%d'%order,[b[0]-np.sqrt(R),b[-1]-np.sqrt(R)],2e-13)
        ck.small_all('balanced constraint including both poles, N=%d'%order,b*b*H,2e-9)
        ck.small_all('endpoint scalars preserved, N=%d'%order,[phi[0]-solution.phim,phi[-1]-solution.phip],2e-13)
        ck.true('positive smooth interior warp factor, N=%d'%order,np.isfinite(b).all() and np.min(b)>0)
        ck.true('concavity lower bound on b, N=%d'%order,np.min(b)>=np.sqrt(R)-2e-12)
        ck.large('nonconstant scalar remains, N=%d'%order,np.ptp(phi),.39)
        original=potential_integrals(solution.sample,solution.alpha)
        balanced=potential_integrals(lambda x:balanced_sample(solution,x),solution.alpha)
        ck.small_all('potential integrals equal exact D/2, N=%d'%order,np.array(original)-DELTA/2,3e-10)
        ck.small_all('potential integrals invariant under balancing, N=%d'%order,np.array(balanced)-original,3e-10)
        Kbal=DELTA/R
        charges=np.array([balanced[0],-balanced[1]])/Kbal
        ck.small_all('continued charges agree with exact formula, N=%d'%order,charges-np.array([R/2,-R/2]),1e-9)
        Koriginal=DELTA/2*(solution.bp**-2+solution.bm**-2)
        ck.large('wrong unbalanced surface gravity denominator is detected, N=%d'%order,
                 balanced[0]/Koriginal-R/2,.1)
        Snh=np.pi*R;Sspherical=2*np.pi*abs(charges[0]*charges[1])
        ck.close('entropy duality using independently integrated charges, N=%d'%order,Sspherical,Snh**2/(2*np.pi),8e-9)
        ck.true('warped entropy exceeds spherical entropy at the same charges, N=%d'%order,Snh>Sspherical)
        # Remove polar coordinate singularities from the proper meridian length.
        def meridian(theta):
            radius=(RP+RM)/2+DELTA*np.cos(theta)/2
            return float(balanced_sample(solution,radius)[0])
        length,error=quad(meridian,0,np.pi,epsabs=2e-10,epsrel=2e-10,limit=300)
        ck.true('meridian is longer than equal-area round sphere, N=%d'%order,
                np.isfinite([length,error]).all() and error<1e-8 and length>np.pi*np.sqrt(R))
        rec=dict(order=order,cutoff=cutoff,alpha=solution.alpha,R=R,normalized_product=solution.product,
                 potential_integrals_original=original,potential_integrals_balanced=balanced,
                 continued_charges=charges.tolist(),balanced_surface_gravity_difference=Kbal,
                 unbalanced_surface_gravity_difference=Koriginal,entropy_near_horizon=Snh,
                 entropy_spherical_same_charges=Sspherical,relative_entropy_excess=Snh/Sspherical-1,
                 meridian_length=length,round_meridian_length=np.pi*np.sqrt(R),
                 maximum_balanced_constraint=float(np.max(np.abs(b*b*H))))
        records.append(rec)
    # Exercise guards against silent domain/solver failures.
    from dataclasses import replace
    from types import SimpleNamespace
    for label,candidate,point in [('outside interval',solution,RP+1e-8),('NaN radius',solution,np.nan),
                                  ('negative inner radius',replace(solution,bm=-solution.bm),.5),
                                  ('failed solver',replace(solution,collocation=SimpleNamespace(success=False,status=1)),.5)]:
        try:balanced_sample(candidate,point)
        except BVPError:ck.true('reject '+label,True)
        else:ck.true('reject '+label,False)
    ck.small_all('continued observables stable under independent endpoint orders',
                 [records[1][key]-records[0][key] for key in ['R','entropy_near_horizon','entropy_spherical_same_charges','meridian_length']],2e-8)
    target=Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1]/"results/data"))/"continuation"
    target.mkdir(parents=True,exist_ok=True)
    (target/'continued_bvp.json').write_text(json.dumps(dict(records=records,
        caveat='Empirical BVP and quadrature diagnostics, not rigorous existence or interval error bounds.'),indent=2)+'\n')
    ck.finish()

if __name__=='__main__':run_checks()
