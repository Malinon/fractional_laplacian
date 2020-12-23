import numpy as np
import pandas as pd
from plotnine import ggplot, aes, geom_point, geom_line
from Fractional_laplacian_aproxinmation import FractionalLaplacianAproximation
from utils_func import kahan_sum
from utils_func import GFunction


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