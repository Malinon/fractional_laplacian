from Fractional_laplacian_linary import FractionalLaplacianAproximationLinary
from Fractional_laplacian_quad import FractionalLaplacianAproximationQuad
from utils_func_essential import kahan_sum
import math
import scipy.special as ss
from utils_func_essential import calculate_c_alpha_1


def calculators_gen_linear(alpha, L, hs, func):
    print(hs)
    return [FractionalLaplacianAproximationLinary(alpha = alpha, L=L,
        func = func, sum_method =kahan_sum, h = h) for h in hs ]

def calculators_gen_quad(alpha, L, hs, func):
    return [FractionalLaplacianAproximationQuad(alpha = alpha, L=L,
        func = func, sum_method =kahan_sum, h = h) for h in hs ]


def calculate_part_II(alpha, func, L, x):
    return func(x) * 2.0 * calculate_c_alpha_1(alpha) / (alpha * L)

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