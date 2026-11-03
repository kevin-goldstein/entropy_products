"""One amplitude example and five entropy-product branches on common axes."""
from pathlib import Path
import csv, hashlib, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
import numpy as np
from output_paths import path_label

ROOT=Path(__file__).resolve().parents[1]

def main():
    out=Path(os.environ.get('HOT_ATTRACTORS_OUTPUT',ROOT/'results/data'))
    source=out/'branches/branch_trace.csv'
    if not source.is_file():source=ROOT/'data/generated/branches/branch_trace.csv'
    with source.open() as fp:rows=list(csv.DictReader(fp))
    plt.rcParams.update({'font.family':'STIXGeneral','mathtext.fontset':'stix','font.size':9,
                         'axes.labelsize':11,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
    colors=['#1f5f95','#b75d1b','#218477','#964b88','#697b20']
    fig=plt.figure(figsize=(7.4,5.2))
    top=[fig.add_axes([.105,.655,.365,.27]),
         fig.add_axes([.615,.655,.365,.27])]
    lower=fig.add_axes([.105,.185,.875,.325])
    lower.axhline(1,color='.55',lw=.7,ls=(0,(3,3)))
    for n,color in zip(range(3,8),colors):
        branch=sorted((r for r in rows if int(r['n'])==n),key=lambda r:float(r['epsilon']))
        assert len(branch)==45 and float(branch[-1]['epsilon'])==6.
        assert all(int(r['nodes'])==n for r in branch)
        eps=np.r_[0.,[float(r['epsilon']) for r in branch]]
        nu=np.r_[float(n),[float(r['nu']) for r in branch]]
        eta=np.r_[1.,[float(r['eta']) for r in branch]]
        assert np.all(eta[1:]<1.)
        lower.plot(nu,eta,color=color,lw=1.7)
        lower.plot(nu[0],eta[0],'o',ms=4,mfc='white',mec=color,mew=1.1)
        lower.plot(nu[-1],eta[-1],'o',ms=4,color=color)
        lower.text(n,1.007,rf'$n={n}$',color=color,fontsize=10,
                   ha='center',va='center')
        if n==3:
            for a,y,label in zip(top,[nu,eta],[r'$\nu$',r'$\eta$']):
                a.plot(eps,y,color=color,lw=1.7)
                a.plot(eps[0],y[0],'o',ms=4,mfc='white',mec=color,mew=1.1)
                a.plot(eps[-1],y[-1],'o',ms=4,color=color)
                a.set_xlabel(r'$\phi_+$');a.set_ylabel(label)
                a.set_xlim(-.12,6.15);a.set_xticks([0,2,4,6])
                a.set_title(r'$n=3$',color=color,fontsize=10,pad=6)
            top[0].set_yticks([2.95,2.975,3.]);top[0].set_ylim(2.948,3.003)
            top[1].set_yticks([.965,.98,1.]);top[1].set_ylim(.961,1.002)
    lower.set_xlim(2.87,7.16);lower.set_xticks(range(3,8))
    lower.set_ylim(.885,1.016);lower.set_yticks([.9,.92,.94,.96,.98,1.])
    lower.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    lower.set_xlabel(r'$\nu$')
    lower.set_ylabel(r'$\eta=S_+S_-/S_{\mathrm{ext}}^2$')
    for ax in [*top,lower]:
        ax.spines[['top','right']].set_visible(False)
        ax.grid(axis='y',color='.9',lw=.5);ax.set_axisbelow(True)
    handles=[Line2D([],[],marker='o',mfc='white',mec='.3',color='none',ms=4,label=r'Analytic limit: $\phi_+\to0$'),
             Line2D([],[],marker='o',mfc='.3',mec='.3',color='none',ms=4,label=r'Last sample: $\phi_+=6$')]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.54,.025),ncol=2,frameon=False,fontsize=9)
    (out/'figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(out/'figures/branches.pdf');fig.savefig(out/'figures/branches.png',dpi=250);plt.close(fig)
    (out/'branches').mkdir(parents=True,exist_ok=True)
    (out/'branches/figure_metadata.json').write_text(json.dumps({'source':path_label(source,ROOT),
        'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'samples_per_branch':45,
        'upper_panels':'nu and eta versus phi_plus for n=3 only',
        'lower_panel':'eta versus absolute nu for all n=3,...,7 on common horizontal and vertical axes',
        'zero_amplitude':'Analytic limits, not integrated zero-amplitude data',
        'endpoint':'Finite samples at phi_plus=6, not established endpoints'},indent=2)+'\n')
    print('[PASS] Generated one amplitude example and all five coupling-product branches on common axes')

if __name__=='__main__':main()
