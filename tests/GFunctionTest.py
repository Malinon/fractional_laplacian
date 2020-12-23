import unittest
import numpy as np
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils_func import kahan_sum
from utils_func import GFunction


import math
de =2 
h = 0.1
C_1_alpha = 2.0

class GFunTest(unittest.TestCase):
    def setUp(self):
        self.GAlpha1 = GFunction(1.0, h)
        self.GAlpha1.set_C_alpha_1(C_1_alpha)
        self.GAlpha1Half = GFunction(1.5, h)
        self.GAlpha1Half.set_C_alpha_1(C_1_alpha)
        self.GAlpha08 = GFunction(0.8, h)
        self.GAlpha08.set_C_alpha_1(C_1_alpha)
        self.delta = 0.01
    
    def testGSecond(self):
        self.assertEqual(self.GAlpha1.gen_G_second_derivative_at_1(), -1)
        self.assertEqual(self.GAlpha1Half.gen_G_second_derivative_at_1(), -1)
        self.assertEqual(self.GAlpha08.gen_G_second_derivative_at_1(), -1)
    
    def testG(self):
        g1 = self.GAlpha1.gen_G_fun()
        self.assertEqual(g1(1.0), 1,0)
        self.assertEqual(g1(math.e), 0)
        self.assertEqual(g1(math.e ** 2), -(math.e ** 2))

        g1Half = self.GAlpha1Half.gen_G_fun()
        self.assertAlmostEqual(g1Half(1.0), 2.66667, delta=self.delta)
        self.assertAlmostEqual(g1Half(2.0), 3.771236, delta=self.delta)
        self.assertAlmostEqual(g1Half(3.0), 4.618802154, delta=self.delta)

        g08 = self.GAlpha08.gen_G_fun()
        self.assertAlmostEqual(g08(1.0), -5.208333, delta=self.delta)
        self.assertAlmostEqual(g08(2.0), -11.9656, delta=self.delta)
        self.assertAlmostEqual(g08(3.0), -19.46454, delta=self.delta)
        self.assertAlmostEqual(g08(4.0), -27.4897481, delta=self.delta)
        self.assertAlmostEqual(g08(5.0), -35.930459934, delta=self.delta)
    
    def testGDerivative(self):
        g1 = self.GAlpha1.gen_G_fun_derivative()
        self.assertEqual(g1(1), 0)
        self.assertEqual(g1(math.e), -1)
        self.assertEqual(g1(math.e ** 2), -2)

        g1Half = self.GAlpha1Half.gen_G_fun_derivative()
        self.assertAlmostEqual(g1Half(1.0), 1.333333, delta=self.delta)
        self.assertAlmostEqual(g1Half(2.0), 0.942809042, delta=self.delta)
        self.assertAlmostEqual(g1Half(3.0), 0.769800359, delta=self.delta)
        self.assertAlmostEqual(g1Half(4.0), 0.666666, delta=self.delta)

        g08 = self.GAlpha08.gen_G_fun_derivative()
        self.assertAlmostEqual(g08(1.0), -6.25, delta=self.delta)
        self.assertAlmostEqual(g08(2.0), -7.179364719, delta=self.delta)
        self.assertAlmostEqual(g08(3.0), -7.785818373, delta=self.delta)
        self.assertAlmostEqual(g08(4.0), -8.246924442, delta=self.delta)
        self.assertAlmostEqual(g08(5.0), -8.623310384, delta=self.delta)
    
    def test_kahan(self):
        dat = np.zeros(100)
        self.assertEqual(kahan_sum(dat), 0)
        for i in range(100):
            dat[i] = float(i)
        
        self.assertEqual(kahan_sum(dat), 4950)

if __name__ == '__main__':
    unittest.main()