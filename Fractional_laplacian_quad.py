from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximation
from utils_func import GFunction
import numpy as np

class FractionalLaplacianAproximationQuad(FractionalLaplacianAproximation):

    def __init__(self, alpha, h, L, func, sum_method, dim = 1):
        super(FractionalLaplacianAproximationQuad,self).__init__(self, alpha, h
        , L, func, sum_method,  dim)

        g_func_gen = GFunction(alpha, h)
        g_func_gen.set_C_alpha_1(self.C_ALPHA_1)

        self.G_FUNCTION = g_func_gen.gen_G_fun()
        self.G_DERIVATIVE = g_func_gen.gen_G_fun_derivative()
        self.G_SECOND_DERIVATIVE_AT_1 = g_func_gen.gen_G_second_derivative_at_1()
        self.G_GENERAL_MULTI = g_func_gen.gen_general_multiplication()
        self.sum_method = sum_method

        if self.num_of_steps > 0:
            self.w_j_list = self.calculate_w_j_params()


    def calculate_w_j_params(self):
        w_j_params = np.zeros(int(self.num_of_steps))
        w_j_params[0] = (1.0 / (2.0 - self.ALPHA) - self.G_SECOND_DERIVATIVE_AT_1 -
        (self.G_DERIVATIVE(3.0) + 3 * self.G_DERIVATIVE(1.0)) / 2.0 +
        self.G_FUNCTION(3.0) - self.G_FUNCTION(1.0))

        for j in range(2, int(self.num_of_steps) + 1):
            if j % 2 == 0 :
                w_j_params[j - 1] = 2 * (self.G_DERIVATIVE(j + 1) + self.G_DERIVATIVE(j - 1) -
                self.G_FUNCTION(j + 1) + self.G_FUNCTION(j - 1))
            else:
                w_j_params[j - 1] = -0.5 * (self.G_DERIVATIVE(j + 2) + 6 * self.G_DERIVATIVE(j)
                + self.G_DERIVATIVE(j - 2)) + self.G_FUNCTION(j + 2) - self.G_FUNCTION(j - 2)

        return w_j_params