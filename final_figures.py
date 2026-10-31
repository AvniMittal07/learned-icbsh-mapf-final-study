"""Standard Matplotlib figures generated only from the frozen final analysis."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='results/recovery/final_study/figures')
    args=parser.parse_args();out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    summary=json.loads(Path('results/recovery/final_study/summary.json').read_text())
    assert summary['valid_for_inference']
    policies=summary['policies'];order=['model_b','model_a','pool_random','plain','cbs']
    names={'model_b':'Model B (primary)','model_a':'Model A','pool_random':'Random in pool','plain':'Plain ICBS-h','cbs':'Plain CBS'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,(left,right)=plt.subplots(1,2,figsize=(11.6,4.8),gridspec_kw={'width_ratios':[1.05,1]},layout='constrained')
    for index,name in enumerate(order):
        s=policies[name];value=s['geometric_family_time_ratio'];lo,hi=s['stratified_family_bootstrap_95_interval']
        left.errorbar(value,index,xerr=[[value-lo],[hi-value]],fmt='o',capsize=4,
                      color='#1261a0' if name=='model_b' else '#5b626c',markersize=6)
    left.axvline(1,color='#a0a0a0',ls='--',lw=1)
    left.set_yticks(range(len(order)),[names[n] for n in order]);left.invert_yaxis();left.set_xscale('log')
    left.set_xlabel('Family capped-time ratio to residual\nLower is faster; line = residual parity')
    left.set_title('Final test: paired runtime ratios',fontweight='bold');left.grid(axis='x',alpha=.15)
    colors={'random-32-32-20':'#1261a0','room-32-32-4':'#d17c21','warehouse-32-32-aisles':'#38865c'}
    labels={'random-32-32-20':'Random','room-32-32-4':'Room','warehouse-32-32-aisles':'Warehouse'}
    families=policies['model_b']['family_rows']
    for name,color in colors.items():
        rows=[r for r in families if r['map_name']==name]
        right.scatter([r['baseline_mean_capped_seconds'] for r in rows],[r['policy_mean_capped_seconds'] for r in rows],
                      s=48,c=color,label=labels[name],alpha=.8,edgecolors='white',linewidth=.5)
    right.plot([0,15],[0,15],ls='--',color='#888',lw=1);right.set_xlim(-.3,15.5);right.set_ylim(-.3,15.5)
    right.set_xlabel('Residual: family mean capped seconds');right.set_ylabel('Model B: family mean capped seconds')
    right.set_title('18 independent scenario families',fontweight='bold');right.legend(frameon=False,fontsize=8,loc='upper left')
    right.grid(alpha=.12)
    fig.suptitle('Frozen learned conflict selectors: final evaluation',fontsize=14,fontweight='bold')
    fig.supxlabel('15 s caps; 3 nested agent counts × 2 timing repeats per family.\n95% map-stratified family bootstrap intervals; secondary comparisons are exploratory.',fontsize=8)
    fig.savefig(out/'final_runtime.png',dpi=180);fig.savefig(out/'final_runtime.svg');plt.close(fig)
    print(json.dumps(dict(png=str(out/'final_runtime.png'),svg=str(out/'final_runtime.svg'),matplotlib=matplotlib.__version__)))


if __name__=='__main__':main()
