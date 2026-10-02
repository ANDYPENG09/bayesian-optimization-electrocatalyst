import sys
import tempfile
import unittest
from pathlib import Path
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from import_analyzer_results import join_results

class ImportTests(unittest.TestCase):
    def test_join_preserves_order_units_and_qc(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)
            pd.DataFrame({'sample_id':['b','a'],'pH':[2,1]}).to_csv(p/'design.csv',index=False)
            pd.DataFrame({'sample_id':['a','b'],'ma_0_9V_A_mg':[.4,.5],'qc_status':['Good','Invalid']}).to_csv(p/'results.csv',index=False)
            r=join_results(p/'design.csv',p/'results.csv',p/'out.csv')
            self.assertEqual(r.sample_id.tolist(),['b','a'])
            self.assertEqual(r.ma_0_9V_A_mg.tolist(),[.5,.4])
            self.assertEqual(r.qc_status.tolist(),['Invalid','Good'])
    def test_mismatched_duplicate_and_ambiguous_ids(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d);pd.DataFrame({'sample_id':['a','b'],'pH':[1,2]}).to_csv(p/'design.csv',index=False)
            for frame in [pd.DataFrame({'sample_id':['a','c'],'qc_status':['Good']*2}),pd.DataFrame({'sample_id':['a','a'],'qc_status':['Good']*2}),pd.DataFrame({'sample_id':['a','b'],'pH':[3,4],'qc_status':['Good']*2})]:
                frame.to_csv(p/'results.csv',index=False)
                with self.assertRaises(ValueError):join_results(p/'design.csv',p/'results.csv',p/'out.csv')
            self.assertFalse((p/'out.csv').exists())
