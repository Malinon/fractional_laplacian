import pandas as pd
from functools import reduce
from plotnine import ggplot, aes, geom_point, xlab
from Fractional_laplacian_linary import FractionalLaplacianAproximationLinary
from utils_func_check import calculate_part_II
from utils_func_check import exp_minus_squared


class Proposition44(object):
    def __init__(self):
        pass

    def alphas_to_zero(self, q, n = 15):
        assert(q < 1)
        return [q ** s for s in range(1, n)]


    def calculate_for_alphas(self, alphas, num_of_steps, h, func, points_indexs):
        dict_numeric = {}

        for alpha in alphas:
            calc = FractionalLaplacianAproximationLinary(alpha = alpha, h = h,
            num_of_steps = num_of_steps, func = func)
            dict_numeric[alpha] = [calc.get_value_at(ind) +
            calculate_part_II(alpha=alpha, func = func, L=h * num_of_steps, x = ind * h)
            for ind in points_indexs]
    
        output = pd.DataFrame(data = dict_numeric).transpose()
        output.columns = [idx * h for idx in points_indexs]

        return output
    
    def plot_alphas_to_zero(self, num_of_steps, h, func, points_indexs,
    difference = True, n = 10, q = 0.5, file_name = "Zbieganie.png"):
        alphas = self.alphas_to_zero(q, n)
        frac_laplacians = self.calculate_for_alphas(alphas, num_of_steps, h, func,
        points_indexs)
        if difference:
            res = pd.concat([abs(frac_laplacians[idx * h] - func(h * idx))
            for idx in points_indexs])
        else:
            res = pd.concat([frac_laplacians[idx * h] for idx in points_indexs])
        numes = list(range(1, n, 1)) *len(points_indexs)
        pos = reduce((lambda a, b: a+b),[[str(idx * h)] *(n - 1)  for idx in points_indexs], [])
        processed_df = pd.DataFrame(data = {"alphs": numes, "points": pos, "vals":
        res})
        p = ggplot() + geom_point(data = processed_df, mapping=aes(
            x = "numes", y = "vals", colour = "points")) + xlab(
                "Logarytm o podstawie 0.5 z alfa")
        p.save(file_name)