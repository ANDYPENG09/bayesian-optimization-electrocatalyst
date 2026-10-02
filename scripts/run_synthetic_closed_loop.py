"""Executed algorithm demonstration; no measured catalyst performance."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from bo_pipeline import run


def main(output='demo_output'):
    directory=Path(output);directory.mkdir(parents=True,exist_ok=True)
    config=dict(variables=[dict(name='x',type='real',low=0,high=1)],
                objective=dict(metrics={'synthetic_reward':dict(direction='max',scale=1)}),
                constraints=[dict(name='synthetic_cost',direction='le',threshold=1.85)],
                recommendation=dict(n_candidates=512,min_feasibility_probability=.8))
    path=directory/'campaign.yaml';path.write_text(yaml.safe_dump(config),encoding='utf-8')
    x=np.array([.05,.2,.35,.5,.9,.97])
    history=pd.DataFrame({'x':x,'synthetic_reward':1-(x-.72)**2,'synthetic_cost':1+x})
    records=[]
    for round_index in range(3):
        data=directory/'history.csv';history.to_csv(data,index=False)
        rec=run(str(data),str(path),seed=round_index,out_json=str(directory/f'round-{round_index+1}.json'),
                trace_path=str(directory/f'trace-{round_index+1}.json'))
        x=rec['next_experiment']['x'];reward=1-(x-.72)**2;cost=1+x
        records.append(dict(round=round_index+1,x=x,predicted_reward=rec['predicted_mean'],
                            observed_synthetic_reward=reward,observed_synthetic_cost=cost,
                            forecast_feasibility_probability=rec['feasibility_probability']))
        history.loc[len(history)]=[x,reward,cost]
    history.to_csv(directory/'history.csv',index=False)
    summary=dict(data_origin='Synthetic mathematical demonstration; not real catalyst measurements',
                 objective='1 - (x - 0.72)^2',constraint='1 + x <= 1.85',known_optimum_x=.72,
                 initial_best_feasible_reward=float((1-(np.array([.05,.2,.35,.5])-.72)**2).max()),
                 final_best_feasible_reward=float(history.loc[history.synthetic_cost<=1.85,'synthetic_reward'].max()),
                 rounds=records)
    (directory/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default='demo_output')
    main(parser.parse_args().output)
