from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximationBase
from utils_func_essential import FFunction
from utils_func_essential import kahan_sum
import numpy as np

class FractionalLaplacianAproximationLinary(FractionalLaplacianAproximationBase):

    def __init__(self, alpha, h, num_of_steps, func, sum_method = kahan_sum, dim = 1,
    double_precision = True):
        super().__init__(alpha, h, num_of_steps, func, sum_method, dim, double_precision)


        f_func_gen = FFunction(self.ALPHA, self.H, self.C_ALPHA_1)
        f_der = f_func_gen.gen_F_derivative()

        f_fun = f_func_gen.gen_F_fun()
        if double_precision:
            self.F_FUNCTION = f_fun
            self.F_DERIVATIVE = f_der
        else:
            self.F_FUNCTION = lambda t: f_fun(np.float32(t))
            self.F_DERIVATIVE = lambda t: f_der(np.float32(t))


        self.GENERAL_MULTI = f_func_gen.gen_general_multiplication()
        self.sum_method = sum_method

        if self.num_of_steps > 0:
            self.calculate_w_j_params()


    def calculate_w_j_params(self):
        # C_alpha_1 / (2 - alpha) is counted in singular part
        self.w_j_params[1] = (- self.F_DERIVATIVE(self.CONST_ONE) +
        self.F_FUNCTION(self.CONST_TWO) - self.F_FUNCTION(self.CONST_ONE))

        for j in range(2, self.num_of_steps + 1):
            self.w_j_params[j] = (self.F_FUNCTION(j + 1) - self.CONST_TWO
            * self.F_FUNCTION(j) + self.F_FUNCTION(j - 1))
        
        # Weight on bondary
        #self.w_j_params[self.num_of_steps] = (self.F_DERIVATIVE(self.num_of_steps) +
        # self.F_FUNCTION(self.num_of_steps) - self.F_FUNCTION(self.num_of_steps - 1))

