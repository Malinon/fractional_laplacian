import math

class GFunction:
    # G will be called with only non-negative args, so abs was deleted
    def __init__(self, alpha, h):
        self.ALPHA = alpha
        self.C_1_ALPHA = -1
        self.H = h

    def set_C_alpha_1(self, C_ALPHA_1):
        self.C_1_ALPHA = C_ALPHA_1

    def gen_G_fun(self):
        if self.ALPHA == 1.0:
            return lambda t : t - t * math.log(t)

        return lambda t : (t ** (2.0 - self.ALPHA)) / (
                (2.0 - self.ALPHA) * (self.ALPHA - 1.0) * self.ALPHA)

    def gen_G_fun_derivative(self):
        if self.ALPHA == 1.0:
            return lambda t: -math.log(t)

        return lambda t:  (t ** (1.0 - self.ALPHA)) / (self.ALPHA * (self.ALPHA - 1.0))

    def gen_G_second_derivative_at_1(self):
        return -1

    # All componenets are mupltiplied by C_1_self.ALPHA / h^alpha
    def gen_general_multiplication(self):
        return (self.H ** (-self.ALPHA)) * self.C_1_ALPHA
