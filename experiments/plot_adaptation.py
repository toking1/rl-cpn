from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SCENES = []
OUT = None
BEST = {'cascading': .50, 'throughput_shock': .55, 'energy_scarcity': .40, 'faulted_recovery': .25}
LABEL = {'full': 'PBARWS', 'no_load': 'Without load branch', 'no_health': 'Without health branch',
         'proportional': 'Proportional baseline', 'fixed_selected': 'Previously selected fixed alpha'}

def plot(records,trajectories,phases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                        'axes.spines.right':False,'pdf.fonttype':42,'savefig.bbox':'tight'})
    colors={'full':'#275d9f','proportional':'#c06a2e','no_load':'#3c8e6b','no_health':'#a45280','fixed_selected':'#666666'}
    for scene in SCENES:
        first_seed = min(r['seed'] for r in records if r['scenario'] == scene and r['strategy'] == 'full')
        typical={r['strategy']:trajectories[r['id']] for r in records if r['scenario']==scene and r['seed']==first_seed and r['id'] in trajectories}
        if 'full' not in typical:continue
        fig,axs=plt.subplots(3,2,figsize=(12,9.5));axes=axs.ravel()
        bounds=[p['start'] for p in typical['full']['phases']]+[typical['full']['phases'][-1]['end']]
        for ax in axes[:4]:
            for i,(a,b) in enumerate(zip(bounds,bounds[1:])):
                if i%2:ax.axvspan(a,b,color='#edf0f3',zorder=0)
                if i:ax.axvline(a,color='.72',linewidth=.6)
            ax.set_xlim(3500,bounds[-1]);ax.set_xlabel('Simulation time (s)')
            ax.grid(axis='y',alpha=.18)
        for strategy in ['full','proportional','no_load','no_health','fixed_selected']:
            if strategy not in typical:continue
            a=np.array(typical[strategy]['grid'])
            axes[0].step(a[:,0],a[:,1],where='post',color=colors[strategy],lw=1.4,
                         label=LABEL[strategy] if strategy!='fixed_selected' else f'Fixed alpha {BEST[scene]:.2f}')
        axes[0].set_ylim(-.03,1.03);axes[0].set_ylabel('Reward weight alpha')
        axes[0].legend(fontsize=7,ncol=2,loc='best',frameon=False)
        a=np.array(typical['full']['grid'])
        axes[1].plot(a[:,0],a[:,2]*100,color=colors['full'],lw=1.2)
        for threshold in [5,15]:axes[1].axhline(threshold,ls='--',color='.5',lw=.7)
        axes[1].set_ylabel('Window rejection rate (%)')
        axes[2].plot(a[:,0],a[:,3]*100,color=colors['full'],lw=1.3,label='Minimum battery')
        for threshold in [30,50,60]:axes[2].axhline(threshold,ls='--',color='.6',lw=.6)
        axes[2].set_ylabel('Minimum battery (%)');axes[2].set_ylim(-3,103)
        availability=axes[2].twinx();availability.step(a[:,0],a[:,4],where='post',color='#b46c32',lw=.9,alpha=.8)
        availability.set_ylabel('Available workers',color='#b46c32');availability.set_ylim(-.1,3.4);availability.set_yticks([0,1,2,3])
        availability.spines['right'].set_visible(True)
        full=[trajectories[r['id']] for r in records if r['scenario']==scene and r['strategy']=='full' and r['id'] in trajectories]
        for detail in full:
            g=np.array(detail['grid']);axes[3].step(g[:,0],g[:,1],where='post',lw=.65,alpha=.3,color=colors['full'])
        if len(full)>1:
            grid=np.array([d['grid'] for d in full]);axes[3].step(grid[0,:,0],grid[:,:,1].mean(axis=0),where='post',color=colors['full'],lw=1.7)
        axes[3].set_ylim(-.03,1.03);axes[3].set_ylabel(f'PBARWS alpha ({len(full)} seeds)')
        for ax,key,ylabel in [(axes[4],'gamma','Timely periodic completion (%)'),(axes[5],'rejection_rate','Phase rejection rate (%)')]:
            x=np.arange(len(bounds)-1);width=.24
            for offset,strategy in enumerate(['full','proportional','fixed_selected']):
                rows=sorted([r for r in phases if r['scenario']==scene and r['strategy']==strategy],key=lambda r:r['start'])
                if len(rows)!=len(x):continue
                means=[r[key+'_mean']*100 for r in rows];errors=[(r[key+'_sd'] or 0)*100 for r in rows]
                ax.bar(x+(offset-1)*width,means,width,yerr=errors,capsize=2,color=colors[strategy],label=LABEL[strategy],error_kw={'lw':.7})
            ax.set_xticks(x,[f'P{i+1}' for i in x]);ax.set_ylabel(ylabel);ax.set_xlabel('Disturbance phase')
            ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
            if key=='gamma':ax.legend(fontsize=6.5,frameon=False)
        for i,ax in enumerate(axes):ax.text(.5,-.25,f'({chr(97+i)})',transform=ax.transAxes,ha='center',va='top',fontsize=10)
        fig.subplots_adjust(hspace=.65,wspace=.42,bottom=.08,top=.96)
        fig.savefig(OUT/f'adaptation_{scene}.pdf')
        plt.close(fig)
    print(json.dumps({'runs_analyzed':len(records),'output':str(OUT.relative_to(REPO)),'figures':len(list(OUT.glob('adaptation_*.pdf')))}))



def main(kind):
    global SCENES, OUT
    parser = argparse.ArgumentParser(description="Plot recorded adaptation trajectories and phase metrics")
    parser.add_argument("--output", type=Path, required=True, help="Experiment output directory under results/")
    args = parser.parse_args()
    if args.output.is_absolute() or ".." in args.output.parts:
        parser.error("--output must be relative to the repository root")
    output = (REPO / args.output).resolve()
    results = (REPO / "results").resolve()
    if not output.is_relative_to(results) or output == results:
        parser.error("--output must be a subdirectory of results/")
    if kind == "ablation":
        SCENES = ["cascading", "throughput_shock", "energy_scarcity", "faulted_recovery"]
        names = ["full", "proportional", "no_load", "no_health", "fixed_selected"]
        aliases = {name: name for name in names}
    elif kind == "extension":
        SCENES = ["cascading", "throughput_shock"]
        aliases = {name + "__w3_wireless": name for name in ["full", "proportional", "fixed_selected"]}
    else:
        raise ValueError("Unknown experiment type")
    records = []
    trajectories = {}
    for path in sorted((output / "analysis/runs").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("scenario") not in SCENES or record.get("strategy") not in aliases:
            continue
        if record.get("status") != "completed":
            raise ValueError(f"Incomplete run: {record.get('id')}")
        detail_path = output / "analysis/trajectories" / (record["id"] + ".json")
        if not detail_path.is_file():
            raise ValueError(f"Missing trajectory for {record['id']}")
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        if not detail.get("phases") or not detail.get("grid"):
            raise ValueError(f"Missing phase or grid data for {record['id']}")
        trajectories[record["id"]] = detail
        record["strategy"] = aliases[record["strategy"]]
        records.append(record)
    if not records:
        raise ValueError("No completed runs with adaptation trajectories were found")
    seeds = sorted({r["seed"] for r in records})
    expected = {(scene, seed, name) for scene in SCENES for seed in seeds for name in aliases.values()}
    actual = [(r["scenario"], r["seed"], r["strategy"]) for r in records]
    if len(actual) != len(expected) or set(actual) != expected:
        raise ValueError("Incomplete or duplicate scenario, seed, and strategy matrix")
    grouped = defaultdict(list)
    for record in records:
        for phase in trajectories[record["id"]]["phases"]:
            grouped[(record["scenario"], record["strategy"], phase["start"], phase["end"])].append(phase)
    phases = []
    for (scene, strategy, start, end), values in grouped.items():
        row = {"scenario": scene, "strategy": strategy, "start": start, "end": end}
        for key in ("gamma", "rejection_rate"):
            samples = [v[key] for v in values if v[key] is not None]
            if len(samples) != len(values):
                raise ValueError(f"Missing {key} in {scene} / {strategy} / {start}")
            row[key + "_mean"] = statistics.mean(samples)
            row[key + "_sd"] = statistics.stdev(samples) if len(samples) > 1 else 0.0
        phases.append(row)
    OUT = output / "plots" / "figures"
    OUT.mkdir(parents=True, exist_ok=True)
    plot(records, trajectories, phases)
