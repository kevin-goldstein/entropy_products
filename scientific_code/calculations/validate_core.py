"""Core regression guards and cross-method comparisons.

Expected-number comparisons are regressions, not independent derivations."""
import json
from pathlib import Path
import os
from fractions import Fraction
import numpy as np
import quartic
import half_interval_scan as half
import orbit_scan as orbit

checks=[]
def check(name,condition):
    if not condition:raise AssertionError(name)
    checks.append(name)

exact=quartic.n3_quartic()
check('Regression: exact quartic product deficit',exact['deficit_quartic']==Fraction(96,1001))
check('Regression: exact quadratic tuning coefficient',exact['nu_quadratic']==-Fraction(96,1001))
row=half.tuned(.2,cutoff=1e-7,order=5,rtol=3e-14)
check('Regression: half-interval coupling',abs(row['alpha']-4.89376036208404)<1e-12)
check('Regression: half-interval entropy product',abs(row['eta']-.999859966473997)<1e-12)
check('Independent reflection and center matching',max(row['center_match'],row['symmetry_phi'])<1e-10)
orbit.configure_seed(row['alpha'],row['bminus'],source='fresh half-interval solution')
ref=orbit.get_reference();comp=orbit.get_compact(ref)
bc=orbit.BREF*np.exp(-orbit.D*comp.t_events[0][0]/2)
check('Regression: compactified exterior endpoint',abs(bc-4.2039691758024)<1e-9)
physical=orbit.make_data(ref,comp,orbit.BREF)
check('Regression: normalized reference mass',abs(physical['M']-5.516147869)<1e-7)
check('Regression: normalized reference modulus',abs(physical['phi_inf']-.975450693973)<1e-9)
check('Signed two-horizon temperature identity',abs(physical['Tplus']*physical['Splus']+physical['Tminus']*physical['Sminus'])<1e-12)
for B in [bc*(1+1e-8),bc*(1+1e-6),orbit.BREF,1000]:
    r=orbit.make_data(ref,comp,B)
    check('Positive finite physical data at B=%g'%B,r['M']>0 and r['c']>0 and r['Tplus']>0 and r['Tminus']<0)
try:orbit.make_data(ref,comp,bc*.99)
except ValueError:rejected=True
else:rejected=False
check('Rejects exterior extrapolation past runaway',rejected)
report={'passed':len(checks),'failed':0,'checks':checks,'half_interval':row,'Bcrit':float(bc)}
out=Path(os.environ.get("HOT_ATTRACTORS_OUTPUT", Path(__file__).resolve().parents[1]/"results/data"))/"branch"
out.mkdir(parents=True,exist_ok=True)
(out/"core_validation.json").write_text(json.dumps(report,indent=2)+"\n")
print(json.dumps(report,indent=2))
