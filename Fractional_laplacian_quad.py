from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximationBase
from utils_func_essential import GFunction
import numpy as np

class FractionalLaplacianAproximationQuad(FractionalLaplacianAproximationBase):

    def __init__(self, alpha, h, num_of_steps, func, sum_method, dim = 1, 
    double_precision = True):
        super().__init__(alpha, h, num_of_steps, func, sum_method, dim, double_precision)

        g_func_gen = GFunction(self.ALPHA, self.H, self.C_ALPHA_1)

        G_FUNCTION = g_func_gen.gen_G_fun()
        G_DERIVATIVE = g_func_gen.gen_G_fun_derivative()
        if double_precision:
            self.G_FUNCTION = G_FUNCTION
            self.G_DERIVATIVE = G_DERIVATIVE
            self.CONST_THREE = 3.0
            self.CONST_SIX = 6.0
        else:
            self.G_FUNCTION = lambda t : G_FUNCTION(np.float32(t))
            self.G_DERIVATIVE = lambda t : G_FUNCTION(np.float32(t))
            self.CONST_THREE = np.float32(3.0)
            self.CONST_SIX = np.float32(6.0)

        self.G_SECOND_DERIVATIVE_AT_1 = g_func_gen.gen_G_second_derivative_at_1()
        self.GENERAL_MULTI = g_func_gen.gen_general_multiplication()
        self.sum_method = sum_method

        self.calculate_w_j_params()


    def calculate_w_j_params(self):
        self.w_j_params[1] = (self.CONST_ONE / (self.CONST_TWO - self.ALPHA)
        - self.G_SECOND_DERIVATIVE_AT_1 - (self.G_DERIVATIVE(self.CONST_THREE)
        + self.CONST_THREE * self.G_DERIVATIVE(self.CONST_ONE)) / self.CONST_TWO +
        self.G_FUNCTION(self.CONST_THREE) - self.G_FUNCTION(self.CONST_ONE))

        for j in range(2, self.num_of_steps + 1):
            if j % 2 == 0 :
                self.w_j_params[j] = self.CONST_TWO * (self.G_DERIVATIVE(j + 1)
                + self.G_DERIVATIVE(j - 1) - self.G_FUNCTION(j + 1) +
                self.G_FUNCTION(j - 1))
            else:
                self.w_j_params[j] = (-(self.G_DERIVATIVE(j + 2)  +
                self.CONST_SIX * self.G_DERIVATIVE(j) +
                self.G_DERIVATIVE(j - 2)) / self.CONST_TWO
                + self.G_FUNCTION(j + 2) - self.G_FUNCTION(j - 2))