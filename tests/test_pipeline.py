import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from bo_pipeline import run, build_space, fit_gp, candidate_pool, acquisition_values, assess_convergence

class PipelineTests(unittest.TestCase):
    def campaign(self, directory, mixed=False):
        variables = [dict(name='x', type='real', low=0, high=1)]
        df = pd.DataFrame({'x': [.05,.2,.4,.6,.8,.95], 'reward': [-.42,-.25,-.09,-.01,-.01,-.0625], 'loss': [.05]*6})
        if mixed:
            variables += [dict(name='hold', type='int', low=1, high=4),dict(name='gas', type='cat', categories=['Ar','H2/Ar'])]
            df['hold'], df['gas'] = [1,2,3,4,1,2], ['Ar','H2/Ar']*3
        config = dict(variables=variables, objective=dict(metrics={'reward':dict(direction='max',scale=1)}),
                      constraints=[dict(name='x',direction='le',threshold=.9),dict(name='loss',direction='le',threshold=.1)],
                      recommendation=dict(n_candidates=256, batch_distance=.01, min_feasibility_probability=.8))
        return df, config

    def execute(self, directory, df, config, **kwargs):
        data, cfg, out = Path(directory)/'data.csv', Path(directory)/'config.yaml', Path(directory)/'out.json'
        df.to_csv(data,index=False)
        cfg.write_text(yaml.safe_dump(config),encoding='utf-8')
        return run(str(data),str(cfg),out_json=str(out),trace_path=str(Path(directory)/'trace.json'),**kwargs)

    def test_batch_unique_reproducible_and_constrained(self):
        with tempfile.TemporaryDirectory() as d:
            df, cfg = self.campaign(d)
            a = self.execute(d,df,cfg,batch=4,seed=7)
            b = self.execute(d,df,cfg,batch=4,seed=7)
            points = [r['parameters']['x'] for r in a['recommendations']]
            self.assertEqual(points,[r['parameters']['x'] for r in b['recommendations']])
            self.assertEqual(len(points),len(set(points)))
            for rec in a['recommendations']:
                self.assertLessEqual(rec['parameters']['x'],.9)
                self.assertGreaterEqual(rec['feasibility_probability'],.8)
                self.assertNotIn(rec['parameters']['x'],df.x.tolist())
                self.assertAlmostEqual(rec['confidence_95'][1]-rec['predicted_mean'],1.96*rec['predicted_std'])
            self.assertEqual(len(a['batch']),3)
            self.assertTrue(all(s['status']=='ok' for s in a['trace']['spans']))
            persisted=json.loads((Path(d)/'out.json').read_text())
            self.assertEqual(persisted['trace'],json.loads((Path(d)/'trace.json').read_text()))

    def test_integer_and_category_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            df,cfg=self.campaign(d,mixed=True)
            rec=self.execute(d,df,cfg,batch=3)
            for r in rec['recommendations']:
                self.assertIsInstance(r['parameters']['hold'],int)
                self.assertIn(r['parameters']['gas'],['Ar','H2/Ar'])
                self.assertTrue(np.isfinite(r['predicted_std']))

    def test_pending_excluded(self):
        with tempfile.TemporaryDirectory() as d:
            df,cfg=self.campaign(d)
            first=self.execute(d,df,cfg)
            cfg['pending_experiments']=[first['next_experiment']]
            second=self.execute(d,df,cfg)
            self.assertNotEqual(first['next_experiment'],second['next_experiment'])

    def test_qc_and_invalid_values(self):
        with tempfile.TemporaryDirectory() as d:
            df,cfg=self.campaign(d)
            df['qc_status']=['Good']*5+['Invalid']
            df.loc[5,'reward']=np.nan
            rec=self.execute(d,df,cfg)
            self.assertEqual(rec['excluded_qc_rows'],1)
            df.loc[0,'reward']=np.inf
            with self.assertRaises(ValueError):self.execute(d,df,cfg)
            cfg['accepted_qc']=['Invalid']
            with self.assertRaises(ValueError):self.execute(d,df,cfg)

    def test_no_feasible_candidate_writes_no_output(self):
        with tempfile.TemporaryDirectory() as d:
            df,cfg=self.campaign(d)
            cfg['constraints'][0]['threshold']=-1
            with self.assertRaisesRegex(ValueError,'No untested candidates'):self.execute(d,df,cfg)
            self.assertFalse((Path(d)/'out.json').exists())

    def test_exhausted_discrete_space_and_bad_batch(self):
        with tempfile.TemporaryDirectory() as d:
            df=pd.DataFrame({'x':[1,2,3],'reward':[0,1,0]})
            cfg=dict(variables=[dict(name='x',type='int',low=1,high=3)],objective=dict(metrics={'reward':{}}))
            with self.assertRaisesRegex(ValueError,'No untested candidates'):self.execute(d,df,cfg)
            with self.assertRaises(ValueError):self.execute(d,df,cfg,batch=0)
            df['x']=df['x'].astype(float)
            df.loc[0,'x']=1.5
            with self.assertRaises(ValueError):self.execute(d,df,cfg)

    def test_direction_and_zero_variance_acquisition(self):
        score, ei=acquisition_values(np.array([0.,1.,2.]),np.zeros(3),1.,'EI',xi=0)
        np.testing.assert_allclose(score,[0,0,1])
        self.assertFalse(assess_convergence([1]*10,1).converged)
        self.assertTrue(assess_convergence([1]*10,0).converged)
        self.assertIsNone(assess_convergence([],0).best_y)

if __name__=='__main__':unittest.main()
