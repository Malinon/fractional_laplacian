import unittest
import os
import sys
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from utils_func import kahan_sum
from utils_func import GFunction
from utils_func import exp_minus_squared
from utils_func import exact_solution_exp_minus_squared_at_0
from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximation

class Test_6_1(unittest.TestCase):
    def setUp(self):
        self.alpha = 0.8
        self.computer =  FractionalLaplacianAproximation(
            self.alpha, h=0.01, L=10.0, func=exp_minus_squared,
            sum_method = kahan_sum, g_func_gen=GFunction(self.alpha , 0.01))

    def test_non_negative_coef(self):
        coef = self.computer.w_j_list
        index = 0
        for w in coef:
            self.assertGreater(w, 0.0, msg=index)
            index = index + 1

    def test_is_like_on_plot(self):
        exact_solution = exact_solution_exp_minus_squared_at_0(self.alpha)

        self.assertAlmostEqual(self.computer.get_value_at(0), exact_solution, delta=0.15)

if __name__ == '__main__':
    unittest.main()
