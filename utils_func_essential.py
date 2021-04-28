import math
import numpy as np
import scipy.special as special


def calculate_c_alpha_1( alpha, dim = 1, double_precision = True):
    if double_precision:
        ONE = 1.0
        TWO = 2.0
        PI = math.pi
    else:
        ONE = np.float32(1.0)
        TWO = np.float32(2.0)
        PI = np.float32(math.pi)
    return (alpha * special.gamma( (alpha + dim) / TWO ) * np.power(
            TWO, (alpha - ONE))) / (np.power(PI, (dim / TWO))
            * special.gamma((TWO - alpha) / TWO))

def kahan_sum(components):
    sum_val = components[0]
    correct_val = sum_val - sum_val # Because of type
    for i in range(1, len(components)):
        y = components[i] - correct_val
        t = sum_val + y
        correct_val = (t - sum_val) - y
        sum_val = t

    return sum_val

class FFunction:
    # G will be called with only non-negative args, so abs was deleted
    def __init__(self, alpha, h, double_precision = True):
        if double_precision:
            self.CONST_ONE = 1.0
            self.CONST_TWO = 2.0
        else:
            self.CONST_ONE = np.float32(1)
            self.CONST_TWO = np.float32(2)
        self.ALPHA = alpha
        self.C_1_ALPHA = -self.CONST_ONE
        self.H = h

    def set_C_alpha_1(self, C_ALPHA_1):
        self.C_1_ALPHA = C_ALPHA_1

    def gen_F_fun(self):
        if self.ALPHA == self.CONST_ONE:
            return lambda t : -np.log(t)

        return lambda t : np.power(t, (self.CONST_ONE - self.ALPHA)) / (
                (self.ALPHA - self.CONST_ONE) * self.ALPHA)

    def gen_F_derivative_at_1(self):
        return -self.CONST_ONE / self.ALPHA

    # All componenets are mupltiplied by C_1_self.ALPHA / h^alpha
    def gen_general_multiplication(self):
        return np.power(self.H, (-self.ALPHA)) * self.C_1_ALPHA

class GFunction:
    # G will be called with only non-negative args, so abs was deleted
    def __init__(self, alpha, h, double_precision = True):
        if double_precision:
            self.CONST_ONE = 1.0
            self.CONST_TWO = 2.0
        else:
            self.CONST_ONE = np.float32(1)
            self.CONST_TWO = np.float32(2)
        self.ALPHA = alpha
        self.C_1_ALPHA = -self.CONST_ONE
        self.H = h

    def set_C_alpha_1(self, C_ALPHA_1):
        self.C_1_ALPHA = C_ALPHA_1

    def gen_G_fun(self):
        if self.ALPHA == self.CONST_ONE:
            return lambda t : t - t * np.log(t)

        return lambda t :np.power(t ,(self.CONST_TWO - self.ALPHA)) / (
                (self.CONST_TWO - self.ALPHA) * (self.ALPHA - self.CONST_ONE) * self.ALPHA)

    def gen_G_fun_derivative(self):
        if self.ALPHA == self.CONST_ONE:
            return lambda t: -np.log(t)

        return lambda t:  np.power(t, (self.CONST_ONE - self.ALPHA)) / (self.ALPHA *
        (self.ALPHA - self.CONST_ONE))

    def gen_G_second_derivative_at_1(self):
        return -self.CONST_ONE / self.ALPHA

    # All componenets are mupltiplied by C_1_self.ALPHA / h^alpha
    def gen_general_multiplication(self):
        return np.power(self.H, (-self.ALPHA)) * self.C_1_ALPHA
