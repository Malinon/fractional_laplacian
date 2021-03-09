from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximation
from utils_func import FFunction
import numpy as np

class FractionalLaplacianAproximationLinary(FractionalLaplacianAproximation):

    def __init__(self, alpha, h, L, func, sum_method, dim = 1):
        super(FractionalLaplacianAproximationLinary,self).__init__(self, alpha, h
        , L, func, sum_method,  dim)


        f_func_gen = FFunction(alpha, h)
        f_func_gen.set_C_alpha_1(self.C_ALPHA_1)

        self.F_FUNCTION = f_func_gen.gen_F_fun()
        self.F_DERIVATIVE_AT_1 = f_func_gen.gen_F_fun_derivative_at_1()
        self.F_GENERAL_MULTI = f_func_gen.gen_general_multiplication()
        self.sum_method = sum_method

        if self.num_of_steps > 0:
            self.w_j_list = self.calculate_w_j_params()


    def calculate_w_j_params(self):
        w_j_params = np.zeros(int(self.num_of_steps))
        w_j_params[0] = (1.0 / (2.0 - self.alpha) - self.F_DERIVATIVE_AT_1 +
        self.F_FUNCTION(2) + self.F_FUNCTION(1))

        for j in range(2, int(self.num_of_steps) + 1):
            w_j_params[j - 1] = (self.F_FUNCTION(j + 1) - 2 * self.F_FUNCTION(j)
            + self.F_FUNCTION(j - 1))  

        return w_j_params