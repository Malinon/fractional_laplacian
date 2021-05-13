from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximationBase
from utils_func_essential import kahan_sum
from utils_func_essential import GFunction
import numpy as np

class FractionalLaplacianAproximationQuad(FractionalLaplacianAproximationBase):

    def __init__(self, alpha, h, num_of_steps, func, sum_method = kahan_sum, dim = 1, 
    double_precision = True):
        super().__init__(alpha, h, num_of_steps, func, sum_method, dim, double_precision)

        g_func_gen = GFunction(self.ALPHA, self.H, self.C_ALPHA_1)

        G_FUNCTION = g_func_gen.gen_G_fun()
        G_DERIVATIVE = g_func_gen.gen_G_fun_derivative()
        G_SEC_DER = g_func_gen.gen_G_second_derivative()
        if double_precision:
            self.G_FUNCTION = G_FUNCTION
            self.G_DERIVATIVE = G_DERIVATIVE
            self.CONST_THREE = 3.0
            self.CONST_SIX = 6.0
            self.G_SECOND_DERIVATIVE = G_SEC_DER
        else:
            self.G_FUNCTION = lambda t : G_FUNCTION(np.float32(t))
            self.G_DERIVATIVE = lambda t : G_FUNCTION(np.float32(t))
            self.CONST_THREE = np.float32(3.0)
            self.CONST_SIX = np.float32(6.0)
            self.G_SECOND_DERIVATIVE = lambda t: G_SEC_DER(np.float32(t))

        self.GENERAL_MULTI = g_func_gen.gen_general_multiplication()
        self.sum_method = sum_method

        self.calculate_w_j_params()


    def calculate_w_j_params(self):
        # C_alpha_1 / (2 - alpha) is counted in singular part
        self.w_j_params[1] = (- self.G_SECOND_DERIVATIVE(1) -
        (self.G_DERIVATIVE(self.CONST_THREE)
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
        
        to_cut = - (self.G_FUNCTION(self.num_of_steps) - self.G_FUNCTION(self.num_of_steps + 1)
        + (self.G_FUNCTION(self.num_of_steps) + self.G_FUNCTION(self.num_of_steps + 1))
        / self.CONST_TWO)

        self.w_j_params[self.num_of_steps] = self.w_j_params[self.num_of_steps] - to_cut
        self.w_j_params[self.num_of_steps] = (
            self.G_SECOND_DERIVATIVE(self.num_of_steps)
            - self.CONST_TWO * (-self.G_DERIVATIVE(self.num_of_steps -1)
            + self.G_FUNCTION(self.num_of_steps)
            - self.G_FUNCTION(self.num_of_steps - 1)))