import math
import scipy.special as ss

def kahan_sum(components):
    sum_val = components[0]
    correct_val = 0.0
    for i in range(1, len(components)):
        y = components[i] - correct_val
        t = sum_val + y
        correct_val = (t - sum_val) - y
        sum_val = t

    return sum_val

# Function from 6.1
def exp_minus_squared(x):
    return math.exp(-(x ** 2.0))

# Exact fractional laplacian for #6.1
def exact_solution_exp_minus_squared_at_0(alpha):
    return (2.0 ** alpha) * math.gamma((1.0 + alpha) / 2.0) / math.sqrt(math.pi)

#Analitycal sol Laplacian for exp(-x^2) in 0, but on limited domain
def analitic_sol_fun(x):
    if x == 0:
        return 0
    
    return ss.erf(x) * math.sqrt(math.pi) + (math.exp(-x ** 2.0) - 1.0) / x

#Integral for Laplacian for exp(-x^2) in 0, but on limited domain
def analitic_sol_val(start, end):
    return analitic_sol_fun(end) - analitic_sol_fun(start)

#Laplacian of  (1 + x^2)^((alpha-1)/2)
def fun_2_smooth_lap_accurate(x, alpha):
    return ((2**alpha) * math.gamma((1.0 + alpha) / 2.0)  *
    ((1 + x ** 2) ** (-(alpha + 1) / 2))) / math.gamma((1.0 - alpha) / 2.0)


class FFunction:
    # G will be called with only non-negative args, so abs was deleted
    def __init__(self, alpha, h):
        self.ALPHA = alpha
        self.C_1_ALPHA = -1
        self.H = h

    def set_C_alpha_1(self, C_ALPHA_1):
        self.C_1_ALPHA = C_ALPHA_1

    def gen_F_fun(self):
        if self.ALPHA == 1.0:
            return lambda t : -math.log(t)

        return lambda t : (t ** (1.0 - self.ALPHA)) / (
                (self.ALPHA - 1.0) * self.ALPHA)

    def gen_F_derivative_at_1(self):
        return -1.0 / self.alpha

    # All componenets are mupltiplied by C_1_self.ALPHA / h^alpha
    def gen_general_multiplication(self):
        return (self.H ** (-self.ALPHA)) * self.C_1_ALPHA

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
        return -1.0 / self.ALPHA

    # All componenets are mupltiplied by C_1_self.ALPHA / h^alpha
    def gen_general_multiplication(self):
        return (self.H ** (-self.ALPHA)) * self.C_1_ALPHA
