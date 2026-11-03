"""Entropy deficit figure from existing numerical shooting data.

No ODE solution or physical parameter is fitted here. Markers are the nine
stored numerical deficits; the curve is the leading quartic asymptotic term.
Run with Python plus the existing NumPy/Matplotlib dependencies.
"""
from pathlib import Path
import argparse
import os
import csv
import hashlib
import json
import platform
from output_paths import path_label
from fractions import Fraction
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter

HERE=Path(__file__).resolve().parent
PACKAGE=HERE.parent
OUT=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT',PACKAGE/'results/data'))
SOURCE=OUT/'branch/amplitude_scan.csv'
if not SOURCE.is_file(): SOURCE=PACKAGE/'data/generated/branch/amplitude_scan.csv'
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--input',type=Path,default=SOURCE)
parser.add_argument('--output',type=Path,default=OUT/'figures/deficit.pdf')
args=parser.parse_args()
SOURCE=args.input.resolve();OUTPUT=args.output.resolve()
with SOURCE.open(newline='') as stream:
    rows=list(csv.DictReader(stream))
eps=np.array([float(row['epsilon']) for row in rows])
deficit=np.array([float(row['deficit']) for row in rows])
assert len(rows)==9 and np.all(np.diff(eps)>0)
assert np.all(deficit>0)
assert np.max(np.abs(deficit-(1-np.array([float(row['eta']) for row in rows]))))<2e-16
A=Fraction(96,1001)
a=float(A)
x=np.geomspace(.021,.375,401)
leading=a*x**4

plt.rcParams.update({
    'font.family':'serif','font.serif':['DejaVu Serif'],
    'font.size':8,'axes.labelsize':8,'xtick.labelsize':7,'ytick.labelsize':7,
    'legend.fontsize':7,'mathtext.fontset':'dejavuserif',
    'axes.linewidth':.6,'xtick.major.width':.6,'ytick.major.width':.6,
    'xtick.minor.width':.45,'ytick.minor.width':.45,
    'xtick.major.size':3,'ytick.major.size':3,
    'xtick.minor.size':1.5,'ytick.minor.size':1.5,
    'pdf.fonttype':42,'ps.fonttype':42,
})
fig=plt.figure(figsize=(3.4,2.75))
ax=fig.add_axes([.18,.18,.79,.79])
ax.set_xscale('log');ax.set_yscale('log')
num=ax.plot(eps,deficit,'o',color='#151515',mec='white',mew=.4,ms=3.2,zorder=5,label='Numerical shooting')[0]
q=ax.plot(x,leading,color='#626262',lw=1.1,ls=(0,(4,2.5)),label=r'$(96/1001)\epsilon^4$')[0]
ax.set_xlim(.02,.40);ax.set_ylim(1.1e-8,2.5e-3)
ax.set_xlabel(r'$\epsilon=\phi_+-\phi_0$')
ax.set_ylabel(r'$1-\eta\,,\quad \eta=S_+S_-/S_{\mathrm{ext}}^2$',labelpad=3)
ax.xaxis.set_major_locator(FixedLocator([.025,.05,.1,.2,.35]))
ax.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:g}'))
ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_major_locator(FixedLocator([1e-8,1e-7,1e-6,1e-5,1e-4,1e-3]))
ax.tick_params(which='both',direction='in',top=True,right=True)
ax.legend(handles=[num,q],loc='upper left',frameon=False,handlelength=2.4,
          borderaxespad=.45,labelspacing=.45,handletextpad=.55)

# Scaling out epsilon^4 makes the first correction visible at column width.
inset=ax.inset_axes([.53,.12,.43,.32])
y=np.linspace(0,.125,301)
inset.plot(y,np.full_like(y,a),color='#626262',lw=.85,ls=(0,(4,2.5)))
inset.plot(eps**2,deficit/eps**4,'o',color='#151515',mec='white',mew=.3,ms=2.3,zorder=5)
inset.set_xlim(0,.13);inset.set_ylim(.065,.104)
inset.set_xlabel(r'$\epsilon^2$',fontsize=7,labelpad=.7)
inset.set_ylabel(r'$(1-\eta)/\epsilon^4$',fontsize=7,labelpad=1)
inset.set_xticks([0,.05,.1]);inset.set_xticklabels(['0','0.05','0.10'])
inset.set_yticks([.07,.08,.09]);inset.tick_params(direction='in',labelsize=6,length=2,pad=1.3,top=True,right=True)
inset.text(.97,.98,r'$96/1001\simeq0.0959$',transform=inset.transAxes,ha='right',va='top',fontsize=5.1)
for spine in inset.spines.values():spine.set_linewidth(.5)

OUTPUT.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(OUTPUT,metadata={'Title':'Entropy-product deficit on the tuned family',
                            'Subject':'Numerical shooting versus the leading quartic asymptote',
                            'Creator':'Matplotlib; plot_deficit.py',
                            'CreationDate':None,'ModDate':None})
fig.savefig(OUTPUT.with_suffix('.png'),dpi=300)
plt.close(fig)
record_dir=OUT/'deficit'
record_dir.mkdir(parents=True,exist_ok=True)
with (record_dir/'plotted_data.csv').open('w',newline='') as stream:
    writer=csv.writer(stream)
    writer.writerow(['epsilon','numerical_deficit','quartic_asymptote','numerical_scaled_deficit'])
    for e,d in zip(eps,deficit):writer.writerow([e,d,a*e**4,d/e**4])
provenance={
    'source_snapshot':path_label(SOURCE, PACKAGE),
    'packaged_reference':'data/generated/branch/amplitude_scan.csv',
    'source_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    'numerical_generator':'calculations/half_interval_scan.py',
    'reference_validation':'validation/reference/standard/half_interval_scan.log',
    'new_ODE_solves':False,'numerical_points':len(rows),
    'epsilon_range':[float(eps.min()),float(eps.max())],
    'numerical_settings':{'horizon_cutoff':1e-6,'Taylor_order':5,'relative_tolerance':3e-14,
                          'coordinate_rminus':.35,'coordinate_rplus':1.0},
    'quartic_coefficient':str(A),
    'curve_scope':'Leading quartic asymptote only; not an exact solution or interpolated numerical data.',
    'dimensions_inches':[3.4,2.75],
    'software':{'python':platform.python_version(),'numpy':np.__version__,'matplotlib':matplotlib.__version__},
    'pdf_sha256':hashlib.sha256(OUTPUT.read_bytes()).hexdigest(),
}
(record_dir/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print(json.dumps(provenance,indent=2))
