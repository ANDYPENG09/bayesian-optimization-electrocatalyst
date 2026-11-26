import unittest
import numpy as np
from core import spectrum, plane_candidates, particles, statistics
class TestTEM(unittest.TestCase):
 def test_fft(self):
    y,x=np.indices((256,256)); scale=.025; a=np.cos(2*np.pi*(x*scale/.226))
    _,p=spectrum(a,scale,.15,.3)
    self.assertTrue(p);self.assertAlmostEqual(p[0]['d_nm'],.226,delta=.005)
    self.assertEqual(plane_candidates(p[0]['d_nm'])[0]['hkl'],'(111)')
 def test_particles(self):
    y,x=np.indices((192,192)); a=np.ones_like(x,dtype=float)
    for cy,cx,r in [(40,40,10),(100,80,15),(140,145,20)]:a[(y-cy)**2+(x-cx)**2<=r*r]=0
    _,rows,_=particles(a,.1,background_nm=4,min_nm=1,max_nm=5)
    self.assertEqual(len(rows),3)
    self.assertAlmostEqual(statistics(rows,.1,a.size)['mean_nm'],3,delta=.2)
 def test_dispersion(self):
    r=statistics([{'ecd_nm':3,'x_px':0,'y_px':0}],.1,100)
    self.assertAlmostEqual(r['geometric_metal_dispersion_pct'],100*6*.3923/(np.sqrt(3)*3),places=5)
if __name__=='__main__':unittest.main()
