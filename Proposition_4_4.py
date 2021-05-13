from Fractional_laplacian_quad import FractionalLaplacianAproximationQuad
import pandas as pd
from functools import reduce
from plotnine import ggplot, aes, geom_point, geom_line, xlab, ylab
from utils_func_check import calculate_part_II

POINTS_LABEL = "Punkty"
VALS_LABEL = "Vals"
ALPHAS_LABEL = "alfa"
Y_LAB_DIFF_TRUE = "Odległość"
Y_LAB_DIFF_FALSE = ""

class Proposition44(object):
    def __init__(self):
        pass

    def alphas_to_zero(self, q, n = 15):
        assert(q < 1)
        return [q ** s for s in range(1, n)]

    def alphas_to_two(self, q, n = 15):
        assert(q < 1)
        return [2 - q ** s for s in range(1, n)]


    def calculate_for_alphas(self, alphas, num_of_steps, h, func, points_indexs):
        dict_numeric = {}

        for alpha in alphas:
            calc = FractionalLaplacianAproximationQuad(alpha = alpha, h = h,
            num_of_steps = num_of_steps, func = func)
            dict_numeric[alpha] = [calc.get_value_at(ind) +
            calculate_part_II(alpha=alpha, func = func, L=h * num_of_steps, x = ind * h)
            for ind in points_indexs]
    
        output = pd.DataFrame(data = dict_numeric).transpose()
        output.columns = [idx * h for idx in points_indexs]

        return output
    
    # func = u
    def plot_alphas_to_zero(self, num_of_steps, h, func, points_indexs,
    difference = True, n = 10, q = 0.5, file_name = "Zbieganie.png"):
        self.plot_alphas(num_of_steps, h, func, points_indexs, self.alphas_to_zero,
        difference, n, q, file_name, x_label="Logarytm o podstawie 0.5 z alfa")

    # func = u
    # lim = laplacian u
    def plot_alphas_to_two(self, num_of_steps, h, func, lim, points_indexs,
    difference = True, n = 10, q = 0.5, file_name = "Zbieganie.png"):
        self.plot_alphas(num_of_steps, h, func, points_indexs, self.alphas_to_two,
        difference, n, q, file_name, x_label="Logarytm o podstawie 0.5 z  2 - alfa", lim = lim)

    def plot_alphas(self, num_of_steps, h, func,  points_indexs, alphas_gen,
    difference = True, n = 10, q = 0.5, file_name = "Zbieganie.png", x_label = "", lim = None):
        alphas = alphas_gen(q, n)
        frac_laplacians = self.calculate_for_alphas(alphas, num_of_steps, h, func,
        points_indexs)
        if difference:
            res = pd.concat([abs(frac_laplacians[idx * h] + lim(h * idx))
            for idx in points_indexs])
            y_lab = Y_LAB_DIFF_TRUE
        else:
            res = pd.concat([frac_laplacians[idx * h] for idx in points_indexs])
            y_lab = Y_LAB_DIFF_FALSE
        numes = list(range(1, n, 1)) *len(points_indexs)
        pos = reduce((lambda a, b: a+b),[[str(idx * h)] *(n - 1)  for idx in points_indexs], [])
        processed_df = pd.DataFrame(data = {ALPHAS_LABEL: numes, POINTS_LABEL: pos, VALS_LABEL:
        res})
        p = (ggplot(data = processed_df, mapping=aes(x = ALPHAS_LABEL, y = VALS_LABEL,
        colour = POINTS_LABEL, group = POINTS_LABEL)) + geom_line() + geom_point()
        + ylab(y_lab) + xlab(x_label))
        p.save(file_name)