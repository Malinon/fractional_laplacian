import numpy as np
import pandas as pd
import math
from plotnine import ggplot, aes, geom_point, geom_line
#from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximation
from utils_func_essential import kahan_sum
from utils_func_essential import GFunction

def plot_compare_abs_error_zero(calculator_gen ,correct_fun, alpha, hs, L, file_name):
    correct_val = correct_fun(x=0, alpha=alpha)
    calculators = calculator_gen(alpha, L, hs)
    xs =  [abs(calc.get_value_at(0)  - correct_val) for calc in calculators]
    df = pd.DataFrame(data={'h': [math.log(h) / math.log(10) for h in hs],
    "absolute error": xs})
    p = ggplot() + geom_line(data=df, mapping=aes(df['h'], df['absolute error']))
    p.save(file_name)
"""
def plot_6_1(L_max, h, file_name, fun, anal_sol):
    args = [h * i for i in range(1, int(L_max / h))]
    abs_errors = np.zeros(len(args))
    for i, arg in enumerate(args):
        comp_s = FractionalLaplacianAproximation(1.0,
        h, arg, fun, kahan_sum, GFunction(1.0, h))
        abs_errors[i] = abs(comp_s.get_value_at(0.0)
         / comp_s.C_ALPHA_1- anal_sol(0, arg) * 2.0) / (anal_sol(0, arg) * 2.0)
        if abs_errors[i]  > h:
            print("Allert ", arg)
    
    df = pd.DataFrame(data={'x': args, "y": abs_errors})
    p = ggplot() + geom_line(data=df, mapping=aes(df['x'],df['y']))
    p.save(file_name)

def plot_compare(L, h, alpha, file_name, fun, exact_solution):
    args = np.linspace(-L, L)
    vals = np.zeros(len(args))
    for i, arg in enumerate(args):
        vals[i] = exact_solution(arg, alpha)
    df = pd.DataFrame(data={'x': args, "y": vals})

    computer = FractionalLaplacianAproximation(alpha, h, L, fun, kahan_sum, GFunction(alpha, h))
    args2 = np.arange(-int(L/h), int(L/h), 1.0)
    vals2 =np.zeros(len(args2))

    for i in range(len(args2)):
        vals2[i] = computer.get_value_at(args2[i])
        args2[i] = args2[i] * h

    df2 = pd.DataFrame(data={'x': args2, "y": vals2})
    p = ggplot() + geom_line(data=df, mapping=aes(df['x'],df['y'])) + geom_point(data=df2,
    mapping=aes(df2['x'],df2['y']))
    p.save(file_name)
"""