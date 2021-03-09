import math
import numpy as np


class FractionalLaplacianAproximationBase(object):

    def __init__(self, alpha, h, L, func, sum_method, dim = 1):
        self.H = h
        self.num_of_steps = (L / h) - 1 # Because of float number of steps can be wrong
        self.FUNC = func
        self.ALPHA = alpha
        self.DIM = dim
        self.C_ALPHA_1 = self.calculate_c_alpha_1(alpha, dim)

        self.sum_method = sum_method


    def calculate_c_alpha_1(self, alpha, dim):
        return (alpha * math.gamma( (alpha + dim) / 2.0 ) * (
            2 ** (alpha - 1.0))) / ((math.pi ** (dim / 2.0))
            * math.gamma((2.0 - alpha) / 2.0))

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
            u_i_next - 2.0 * u_i + u_i_prev) / (2.0 - self.ALPHA)
        return singu


    def func_at(self, index_of_point):
        return self.FUNC(index_of_point * self.H)

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

        return self.G_GENERAL_MULTI * (self.sum_method(components_of_sum))