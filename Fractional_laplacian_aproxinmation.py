from utils_func_essential import calculate_c_alpha_1
import numpy as np

class FractionalLaplacianAproximationBase(object):

    def __init__(self, alpha, h, num_of_steps, func, sum_method, dim = 1,
    double_precision = True):
        if double_precision:
            self.CONST_ONE = 1.0
            self.CONST_TWO = 2.0
            self.H = h
            self.ALPHA = alpha
            self.DIM = dim
            self.func_at = lambda index_of_point: (
                self.FUNC(index_of_point * self.H))
        else:
            self.CONST_ONE = np.float32(1)
            self.CONST_TWO = np.float32(2)
            self.H = np.float32(h)
            self.ALPHA = np.float32(alpha)
            self.DIM = np.float32(dim)
            self.func_at = lambda index_of_point: (
                self.FUNC(np.float32(index_of_point * self.H)))
        
        self.num_of_steps = num_of_steps
        self.FUNC = func
        self.C_ALPHA_1 = calculate_c_alpha_1(self.ALPHA, self.DIM, double_precision)

        self.sum_method = sum_method


# Get Fractional Laplacian value at given point
    def get_value_at(self, index_of_point):
        if self.num_of_steps <= 0:
            return self.get_value_singular(index_of_point)
        return (self.get_value_singular(index_of_point)
        + self.get_value_tail(index_of_point))

# Calculate singular part of integral
    def get_value_singular(self, index_of_point):
        u_i = self.FUNC(self.H * index_of_point)
        u_i_next = self.FUNC(self.H * (index_of_point + 1))
        u_i_prev = self.FUNC(self.H * (index_of_point - 1))
        singu =-self.C_ALPHA_1 * (self.H ** (-self.ALPHA)) * (
            u_i_next - self.CONST_TWO * u_i + u_i_prev) / (self.CONST_TWO
            - self.ALPHA)
        return singu


    def get_value_tail(self, index_of_point):
        if self.num_of_steps == 0:
            return 0
        components_of_sum = np.zeros(int(self.num_of_steps) * 2)
        u_i = self.func_at(index_of_point)
        index = 0
        index_val = 1
        for w in self.w_j_list:
            components_of_sum[index] = w * (u_i - self.func_at(index_of_point - index_val))
            index = index + 1
            components_of_sum[index] = w * (u_i - self.func_at(index_of_point + index_val))
            index = index + 1
            index_val = index_val + 1

        return self.GENERAL_MULTI * (self.sum_method(components_of_sum))