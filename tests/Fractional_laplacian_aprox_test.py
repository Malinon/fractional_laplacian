import unittest
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

from utils_func import kahan_sum
from utils_func import GFunction
from utils_func import exp_minus_squared


from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximation

def mock_sum(nums):
    acc = 0.0
    for i in nums:
        acc = acc + i
    
    return acc

class Mock1G(GFunction):
    def gen_G_fun(self):
        return lambda t: 1

    def gen_G_fun_derivative(self):
        return lambda t: 1
    
    def gen_G_second_derivative_at_1(self):
        return -1
    
    def gen_general_multiplication(self):
        return 1


class Fractional_laplacian_test(unittest.TestCase):
    def setUp(self):
        self.GMock1 = Mock1G(1,1)

    def test_singular(self):
        #6.1
        self.computer =  FractionalLaplacianAproximation(
            alpha=0.8, h=0.01, L=10.0, func=exp_minus_squared,
            sum_method = kahan_sum, g_func_gen=self.GMock1)
        singular = self.computer.get_value_singular(0)
        singular = singular / self.computer.C_ALPHA_1
        self.assertAlmostEqual(singular,  0.006634787764, delta=0.001)
    
    def test_coeff_gen(self):
        h = 0.1
        L = 2.0
        frac = FractionalLaplacianAproximation(1, h, L, lambda t: 1, mock_sum, self.GMock1)
        self.C_ALPHA_1 = 1.0
        params = frac.calculate_w_j_params()
        self.assertAlmostEqual(params[0], 0.0)
        self.assertAlmostEqual(params[1], 4.0)
        self.assertAlmostEqual(params[2], -4.0)
        self.assertEqual(len(params), L / h)

if __name__ == '__main__':
    unittest.main()