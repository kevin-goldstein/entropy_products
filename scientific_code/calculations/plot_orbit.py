from pathlib import Path
import sys,csv,os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0,str(Path(__file__).resolve().parent))
import orbit_scan as orbit
ref=orbit.get_reference();comp=orbit.get_compact(ref)
bc=orbit.BREF*np.exp(-orbit.D*comp.t_events[0][0]/2)
xs=np.logspace(-10,2,401)
rows=[orbit.make_data(ref,comp,bc*(1+x)) for x in xs]
out=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT',Path(__file__).resolve().parents[1]/'results/data'))
(out/'figures').mkdir(parents=True,exist_ok=True)
(out/'orbit').mkdir(parents=True,exist_ok=True)
with (out/'orbit'/'orbit_plot_data.csv').open('w') as f:
    w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.labelsize':10,
                     'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'ps.fonttype':42})
fig,axs=plt.subplots(1,2,figsize=(7.0,2.65),layout='constrained')
for ax,key,label in zip(axs,['M','Tplus'],[r'$\widehat M$',r'$\widehat T_+$']):
    ax.plot(xs,[r[key] for r in rows],color='#205c89',lw=1.7)
    for scale in [2.5,3,4]:
        B=scale*orbit.BDE;row=orbit.make_data(ref,comp,B)
        ax.plot((B-bc)/bc,row[key],marker='o',ms=3.7,color='#a44a2a')
    ax.set_xscale('log');ax.set_xlabel(r'$(B-B_{\rm crit})/B_{\rm crit}$')
    ax.set_ylabel(label);ax.grid(True,which='major',alpha=.18)
    ax.spines[['top','right']].set_visible(False)
axs[0].set_yscale('log')
axs[1].set_yscale('log')
axs[1].set_ylim(min(r['Tplus'] for r in rows)*.8,.017)
fig.savefig(out/'figures'/'orbit.pdf')
fig.savefig(out/'figures'/'orbit.png',dpi=180)
plt.close(fig)
