from Fractional_laplacian_linary import FractionalLaplacianAproximationLinary
from Fractional_laplacian_quad import FractionalLaplacianAproximationQuad
from utils_func_essential import kahan_sum
import math
import scipy.special as ss
from utils_func_essential import calculate_c_alpha_1
import pandas as pd


def calculators_gen_linear(alpha, num_of_steps_list, hs, func, double_precision = True):
    return [FractionalLaplacianAproximationLinary(alpha = alpha, num_of_steps=tup[1],
        func = func, sum_method =kahan_sum, h = tup[0],
        double_precision = double_precision) for tup in zip(hs,num_of_steps_list)  ]

def calculators_gen_quad(alpha, num_of_steps_list, hs, func, double_precision = True):
    return [FractionalLaplacianAproximationQuad(alpha = alpha, num_of_steps=tup[1],
        func = func, sum_method =kahan_sum, h = tup[0],
        double_precision = double_precision) for tup in zip(hs,num_of_steps_list)  ]

def calculate_part_II(alpha, func, L, x):
    return func(x) * 2.0 * calculate_c_alpha_1(alpha) / (alpha * (L ** alpha) )

def laplacian_exp_minus_squared(t):
    return math.exp(- (t ** 2)) * 2 * (2 * (t ** 2) - 1)

# TODO: Check
def calculate_part_III(alpha, func, L, x, beta):
    L_w = 2 * L
    c_1_alpha = calculate_c_alpha_1(alpha)
    return (ss.hyp2f1(beta, alpha + beta, alpha + beta + 1, - x / L_w) * func(L) *
    c_1_alpha * (L ** beta) / ((alpha + beta) * (L_w **(alpha + beta))))

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


so = calculate_part_III(alpha = 0.4, func = lambda x: (1 + x**2) ** (-0.3) ,
L = 0.5, beta = 0.6, x = 1)
print(so)

print(calculate_part_II(alpha= 0.4, func = lambda x: (1 + x**2) ** (-0.3), L = 4.0, x=1))