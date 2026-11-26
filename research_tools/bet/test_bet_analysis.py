import csv
import tempfile
import unittest
from pathlib import Path
from bet_analysis import bet, analyze


def uptake(p, vm=100, c=100):
    return vm*c*p/((1-p)*(1+(c-1)*p))


class BetTests(unittest.TestCase):
    def test_analytic_bet(self):
        points = [(p, uptake(p)) for p in [.075,.10,.15,.20,.25]]
        r = bet(points)
        self.assertAlmostEqual(r["Vm_cm3_stp_g"], 100)
        self.assertAlmostEqual(r["C"], 100)
        self.assertAlmostEqual(r["S_m2_g"], 435.35)
        self.assertAlmostEqual(r["R2"], 1)

    def test_negative_intercept(self):
        pts = [(p, p/((1-p)*(.01*p-.0001))) for p in [.075,.1,.15,.2,.25]]
        self.assertEqual(bet(pts)["fit_status"], "FAIL")

    def test_no_p95_interpolation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"test.csv"
            with path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["measurement_id","branch","p_rel","v_cm3_g_stp"])
                for p in [.075,.1,.15,.2,.25,.9499]:
                    w.writerow(["synthetic-1","adsorption",p,uptake(p)])
            self.assertIsNone(analyze(path)["synthetic-1"]["V_liq_exact_p95_cm3_g"])


if __name__ == "__main__":
    unittest.main()
