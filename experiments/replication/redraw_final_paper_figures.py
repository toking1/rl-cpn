"""Compact mean-only paper figures, following the original panel layouts."""
from pathlib import Path
import os
import json
import sys
ROOT=Path(os.environ['RL_CPN_PLOT_ROOT'])
sys.path.insert(0,str(ROOT/'paper-revision-deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MultipleLocator, MaxNLocator
import numpy as np

OUT=ROOT/'figures'
OUT.mkdir(exist_ok=True)
DATA=json.loads((ROOT/'plot_data/revision-statistics.json').read_text(encoding='utf-8'))
SENS=json.loads((ROOT/'plot_data/sensitivity.json').read_text(encoding='utf-8'))
SCENES=['throughput_shock','energy_scarcity','faulted_recovery','cascading']
TITLES=['Throughput shock','Energy scarcity','Fault recovery','Cascading disturbances']
M=['gamma','total_throughput','rejection_rate','avg_latency','min_lifespan','total_service_time','battery_variance']
METRIC_TITLES=['Timely completion','Total throughput','Task rejection','Average latency','First battery depletion','Battery-alive time','Battery variance']
YLAB=[r'$\Gamma$ (%)',r'Tasks ($\times 10^5$)','Rejection (%)','Latency (ms)','Time (s)','Time (s)',r'Variance (Wh$^2$)']
SCALE=[100,1e-5,100,1000,1,1,1]
FIXED=[f'fixed_{a:.2f}' for a in np.arange(0,1.001,.05)]
X=np.arange(0,1.001,.05)
DYN=['reactive_rejection','reactive_battery','reactive_hybrid']
COLOR={'reactive_hybrid':'#8854ad','reactive_rejection':'#df518f','reactive_battery':'#ed8b23'}
STYLE={'reactive_hybrid':'-.','reactive_rejection':':','reactive_battery':'--'}
NAME={'reactive_hybrid':'PBARWS','reactive_rejection':'Rejection','reactive_battery':'Battery'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':10,
                     'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,
                     'axes.linewidth':.65,'lines.linewidth':1.2,'pdf.fonttype':42,
                     'axes.spines.top':True,'axes.spines.right':True})
def val(scene,s,key='score'):
    return DATA[scene]['summaries'][s][key]['mean']
def label(s):
    return NAME[s] if s in NAME else r'$\alpha='+s[6:]+'$'
def save(fig,n):
    fig.savefig(OUT/(n+'.pdf'),bbox_inches='tight',pad_inches=.07)
    plt.close(fig)
    print(n,flush=True)
def axis_metric(ax,i):
    ax.set_title(METRIC_TITLES[i],pad=7)
    ax.text(.5,-.245,f'({chr(97+i)})',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    ax.set(xlim=(-.025,1.025),xlabel=r'$\alpha$',ylabel=YLAB[i])
    ax.xaxis.set_major_locator(MultipleLocator(.2));ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.grid(ls=':',lw=.5,color='#c1c6cb',alpha=.9)
def panel_grid():
    fig,axes=plt.subplots(2,4,figsize=(13.2,6.5))
    fig.subplots_adjust(left=.058,right=.992,bottom=.13,top=.945,wspace=.43,hspace=.65)
    for i in range(7):axis_metric(axes.flat[i],i)
    axes.flat[7].axis('off')
    return fig,axes

# Figure 2: overlay the three scenarios in seven metric panels, as in the original.
fig,axes=panel_grid()
scene_colors=['#287fba','#d94a48','#3a9b53'];markers=['o','s','^'];linestyles=['-','--','-.']
for i in range(7):
    ax=axes.flat[i]
    for j,scene in enumerate(SCENES[:3]):
        y=[val(scene,s,M[i])*SCALE[i] for s in FIXED]
        ax.plot(X,y,color=scene_colors[j],marker=markers[j],ms=2.7,ls=linestyles[j],lw=1.0)
        a=float(DATA[scene]['best_fixed'][6:]);v=val(scene,DATA[scene]['best_fixed'],M[i])*SCALE[i]
        ax.scatter(a,v,s=34,facecolors='none',edgecolors=scene_colors[j],lw=.9,zorder=4)
        for strategy, line_style, marker, marker_x in [
            ('reactive_rejection', ':', 'x', .90),
            ('reactive_battery', '--', '+', .95),
            ('reactive_hybrid', '-.', 'D', 1.0),
        ]:
            h=val(scene,strategy,M[i])*SCALE[i]
            ax.axhline(h,color=scene_colors[j],ls=line_style,lw=.85,alpha=.8)
            ax.plot([marker_x],[h],marker=marker,color=scene_colors[j],
                    ms=4,mew=.9,mfc='white',zorder=5)
legend=[Line2D([0],[0],color=scene_colors[j],marker=markers[j],ls=linestyles[j],ms=4,label=TITLES[j]) for j in range(3)]
legend += [
    Line2D([0],[0],color='#555',marker='o',mfc='white',lw=0,label=r'Best fixed $\alpha^*$'),
    Line2D([0],[0],color='#555',ls=':',marker='x',ms=4,label='Rejection mean'),
    Line2D([0],[0],color='#555',ls='--',marker='+',ms=4,label='Battery mean'),
    Line2D([0],[0],color='#555',ls='-.',marker='D',mfc='white',ms=4,label='PBARWS mean'),
]
axes.flat[7].legend(handles=legend,loc='center',frameon=False,fontsize=9,labelspacing=.95,handlelength=2.8)
axes.flat[7].text(.5,-.12,'Colors: scenarios\nHorizontal lines: dynamic means',
                  transform=axes.flat[7].transAxes,ha='center',va='bottom',fontsize=8,color='#555555')
save(fig,'single_scenarios_metrics')

# Figure 3: retain the accepted ranking layout, remove footer and add panel letters.
fig,axes=plt.subplots(1,3,figsize=(15,9.5))
fig.subplots_adjust(left=.065,right=.995,bottom=.11,top=.95,wspace=.35)
for j,(ax,scene) in enumerate(zip(axes,SCENES[:3])):
    order=DATA[scene]['order'];yy=np.arange(24)
    ax.barh(yy,[val(scene,s) for s in order],xerr=[DATA[scene]['summaries'][s]['score']['sd'] for s in order],
            color=[{'reactive_rejection':'#d88324','reactive_battery':'#779247','reactive_hybrid':'#a13b80'}.get(s,'#678eae') for s in order],capsize=2,error_kw={'linewidth':.8})
    ax.set(yticks=yy,yticklabels=[NAME[s] if s in NAME else 'Fixed '+s[6:] for s in order],xlabel='Gamma-7 score',title=TITLES[j],xlim=(0,100))
    ax.text(.5,-.075,f'({chr(97+j)})',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    ax.invert_yaxis();ax.grid(axis='x',alpha=.17);ax.spines[['top','right']].set_visible(False)
    ax.tick_params(axis='y',labelsize=10)
save(fig,'single_scenarios_strategy_rankings')

# Figure 6: the original seven metrics plus a compact legend in a 2 x 4 grid.
fig,axes=panel_grid()
for i in range(7):
    ax=axes.flat[i]
    ax.plot(X,[val('cascading',s,M[i])*SCALE[i] for s in FIXED],color='#287fba',marker='o',ms=3,lw=1.15)
    for s in DYN:
        h=val('cascading',s,M[i])*SCALE[i]
        ax.axhline(h,color=COLOR[s],ls=STYLE[s],lw=1.1)
legend=[Line2D([0],[0],color='#287fba',marker='o',ms=4,label='Fixed alpha')]+[Line2D([0],[0],color=COLOR[s],ls=STYLE[s],label=NAME[s]) for s in DYN]
axes.flat[7].legend(handles=legend,loc='center',frameon=False,fontsize=10,labelspacing=1.05,handlelength=2.8)
save(fig,'cascading_metrics')

# Figure 7: match the reference's ranking annotations, Pareto marks and lead region.
fig,ax=plt.subplots(figsize=(13.2,6.1))
fig.subplots_adjust(left=.065,right=.99,bottom=.12,top=.93)
y=np.array([val('cascading',s) for s in FIXED])
h=val('cascading','reactive_hybrid')
ax.fill_between(X,30,h,where=y<h,color=COLOR['reactive_hybrid'],alpha=.045,label='PBARWS mean leads')
ax.plot(X,y,color='#1779dc',marker='o',ms=4.5,lw=1.2,label=r'Fixed $\alpha$ sweep')
front=set(DATA['cascading']['pareto_front_on_means'])
offsets=[2.4,-2.7,2.4,-2.7,2.4,-2.7,2.4,-2.7,2.4,-2.7,3.6,3.2,-3.2,-3.0,2.8,-2.8,-3.0,-3.2,-3.0,2.8,2.8]
for i,s in enumerate(FIXED):
    rank=DATA['cascading']['summaries'][s]['rank']
    annotation=str(rank)+('(P)' if s in front else '')
    if s=='fixed_0.50':annotation+='  '+f'{y[i]:.2f}'
    ax.text(X[i],y[i]+offsets[i],annotation,ha='center',va='center',fontsize=7.8,color='#1265ba')
for s in DYN:
    v=val('cascading',s);rank=DATA['cascading']['summaries'][s]['rank']
    ax.axhline(v,color=COLOR[s],ls='-.',lw=1.15)
    mark='(P)' if s in front else ''
    ax.text(.985,v+1.0,f'Rank {rank}{mark}. {NAME[s]}: {v:.2f}',ha='right',va='bottom',fontsize=8.6,color=COLOR[s],bbox=dict(facecolor='white',edgecolor=COLOR[s],linewidth=.55,boxstyle='round,pad=.15'))
for alpha,txt,yy in [(float(DATA['faulted_recovery']['best_fixed'][6:]),'Recovery',51.9),
                     (float(DATA['energy_scarcity']['best_fixed'][6:]),'Scarcity',48.0),
                     (float(DATA['throughput_shock']['best_fixed'][6:]),'Shock',63.5)]:
    s=f'fixed_{alpha:.2f}';v=val('cascading',s)
    ax.scatter(alpha,v,s=52,color='#ce3035',edgecolor='#a5161d',lw=.6,zorder=4)
    ax.text(alpha,yy,txt+r' best $\alpha^*$',ha='center',fontsize=8.5,color='#c0262c',bbox=dict(facecolor='white',edgecolor='#e07878',lw=.5,boxstyle='round,pad=.15'))
ax.scatter([],[],s=38,color='#ce3035',label=r'Single-scenario optimum $\alpha^*$')
ax.plot([],[],color=COLOR['reactive_hybrid'],ls='-.',label='Reactive dynamic strategy')
ax.set(xlim=(-.03,1.03),ylim=(30,102),xlabel=r'$\alpha$',ylabel='Mean Gamma-7 score',title='Gamma-7 Comprehensive Score')
ax.xaxis.set_major_locator(MultipleLocator(.1));ax.grid(ls=':',lw=.55,color='#c1c6cb')
ax.legend(loc='upper left',bbox_to_anchor=(.015,.985),fontsize=9,framealpha=.96)
save(fig,'cascading_fixed_vs_dynamic_scores')

# Figure 8: grouped perturbation ticks, showing the four leading mean-score policies.
fig,axes=plt.subplots(2,2,figsize=(13.2,7.4))
fig.subplots_adjust(left=.055,right=.99,bottom=.16,top=.95,wspace=.17,hspace=.70)
abbr=['TR','TP','RR','AL','ML','ST','BV']
for j,(ax,scene) in enumerate(zip(axes.flat,SCENES)):
    top=DATA[scene]['order'][:4]
    palette=['#287fba','#3a9b53'];fi=0
    for pos,s in enumerate(top):
        if s in COLOR:c=COLOR[s]
        else:c=palette[fi];fi+=1
        ax.plot(np.arange(28),[v['scores'][s] for v in SENS[scene]],color=c,marker=['o','s','^','D'][pos],ms=2.6,lw=1.8 if pos==0 else 1.0,ls='-' if pos==0 else '--',label=label(s))
    for k in range(6):ax.axvline(4*k+3.5,color='#c6cbd0',ls=':',lw=.65)
    ax.set(title=TITLES[j],ylabel='Mean Gamma-7 score',xlim=(-.4,27.4))
    ax.text(.5,-.16,f'({chr(97+j)})',transform=ax.transAxes,ha='center',va='top',fontsize=10)
    ax.set_xticks([k*4+1.5 for k in range(7)],abbr)
    ax.set_xticks(range(28),minor=True)
    ax.tick_params(axis='x',which='major',labelsize=8.5,pad=6,length=0)
    ax.tick_params(axis='x',which='minor',length=2)
    ax.yaxis.set_major_locator(MaxNLocator(5));ax.grid(axis='y',ls=':',lw=.5,color='#c1c6cb')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=4,fontsize=8,frameon=False,columnspacing=1.1,handlelength=2.0)
save(fig,'scenario_weight_sensitivity')
