
from pathlib import Path
import os
import csv, json, math, statistics, sys
from collections import Counter
repo=Path(__file__).resolve().parents[2]
root=Path(os.environ['RL_CPN_PLOT_ROOT'])
root.mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(repo/'code'))
from evaluate_gamma7 import AlphaMetrics, OBJECTIVES, gamma7_score, load_score_reference, normalized_objectives, weights_for
data_dir=root.parent/'analysis'
with (data_dir/'metrics.csv').open(encoding='utf8',newline='') as f:
    rows=list(csv.DictReader(f))
scenes=['throughput_shock','energy_scarcity','faulted_recovery','cascading']
strategies=[f'fixed_{i/20:.2f}' for i in range(21)]+['reactive_rejection','reactive_battery','reactive_hybrid']
seeds=sorted({int(r['seed']) for r in rows})
expected={(seed,scene,strategy) for seed in seeds for scene in scenes for strategy in strategies}
keys=[(int(r['seed']),r['scenario'],r['strategy']) for r in rows]
if not seeds or len(keys)!=len(expected) or set(keys)!=expected or len(set(keys))!=len(expected):
    raise ValueError('Experiment matrix is incomplete or contains duplicates')
metrics=[name for name,_ in OBJECTIVES]
data={}
sensitivity={}
max_error=0.0
for scene in scenes:
    sub=[r for r in rows if r['scenario']==scene]
    reference=load_score_reference(data_dir/'references'/f'{scene}.json')
    summaries={}
    normalized=[]
    for r in sub:
        candidate=AlphaMetrics(
            alpha=float(r['alpha']), gamma=float(r['gamma']), total_throughput=float(r['total_throughput']),
            rejection_rate=float(r['rejection_rate']), avg_latency=float(r['avg_latency']),
            min_lifespan=float(r['min_lifespan']), total_service_time=float(r['total_service_time']),
            battery_variance=float(r['battery_variance']), gamma_by_type=[],
            first_death_gap=float(r['first_death_gap']), max_lifespan=float(r['max_lifespan']),
            mission_end=float(r['mission_end']), database=r['database'])
        predicted=gamma7_score(candidate,reference,scenario=scene)
        max_error=max(max_error,abs(predicted-float(r['score'])))
        normalized.append((r['strategy'],normalized_objectives(candidate,reference)))
    if max_error>1e-7:
        raise ValueError(f'Gamma-7 recomputation error exceeds tolerance: {max_error}')
    for strategy in strategies:
        selected=[r for r in sub if r['strategy']==strategy]
        summaries[strategy]={}
        for key in metrics+['score']:
            values=[float(r[key]) for r in selected]
            if not all(math.isfinite(x) for x in values): raise ValueError('Metrics contain non-finite values')
            summaries[strategy][key]={'mean':statistics.mean(values),'sd':statistics.stdev(values) if len(values)>1 else 0.0}
    order=sorted(strategies,key=lambda s:-summaries[s]['score']['mean'])
    for rank,strategy in enumerate(order,1):summaries[strategy]['rank']=rank
    front=[]
    for strategy in order:
        v=[summaries[strategy][name]['mean']*(1 if maximize else -1) for name,maximize in OBJECTIVES]
        dominated=False
        for other in order:
            w=[summaries[other][name]['mean']*(1 if maximize else -1) for name,maximize in OBJECTIVES]
            if all(a>=b for a,b in zip(w,v)) and any(a>b for a,b in zip(w,v)):
                dominated=True
                break
        if not dominated:front.append(strategy)
    data[scene]={'summaries':summaries,'order':order,
                 'best_fixed':next(s for s in order if s.startswith('fixed_')),
                 'pareto_front_on_means':front}
    changes=[]
    for name in metrics:
        for factor in (0.8,0.9,1.1,1.2):
            weights=weights_for(scene)
            weights[name]*=factor
            total=sum(weights.values())
            weights={k:v/total for k,v in weights.items()}
            scores={}
            for strategy in strategies:
                values=[100*sum(weights[k]*norm[k] for k in metrics) for st,norm in normalized if st==strategy]
                scores[strategy]=statistics.mean(values)
            changes.append({'metric':name,'factor':factor,'scores':scores,
                            'winner':max(scores,key=scores.get)})
    sensitivity[scene]=changes
inputs=root/'plot_data'
inputs.mkdir(exist_ok=True)
(inputs/'revision-statistics.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
(inputs/'sensitivity.json').write_text(json.dumps(sensitivity,ensure_ascii=False,indent=2),encoding='utf8')
(root/'data-validation.json').write_text(json.dumps({
    'records':len(rows),'unique':len(set(keys)),'seeds':seeds,'scenarios':scenes,
    'strategies_per_scenario':len(strategies),'max_score_reconstruction_error':max_error,
    'sensitivity_cases_per_scenario':28,
    'input':'../analysis/metrics.csv',
    'original_revision':'197f82838e72b97b732b12861f19f712df73ef80'
},ensure_ascii=False,indent=2),encoding='utf8')
print('Data validation passed; maximum score error',max_error)
for scene in scenes:print(scene,'best_fixed',data[scene]['best_fixed'],'best',data[scene]['order'][0])
