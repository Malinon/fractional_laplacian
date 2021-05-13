from Proposition_4_4 import Proposition44
from utils_func_check import exp_minus_squared
from utils_func_check import laplacian_exp_minus_squared

prop44 = Proposition44()
#prop44.plot_alphas_to_zero(num_of_steps = 10000, h=0.1,
#func=exp_minus_squared, points_indexs=[0, 5, 10], difference = True)
prop44.plot_alphas_to_two(num_of_steps = 10000, h = 0.1,
func=exp_minus_squared, lim = laplacian_exp_minus_squared, points_indexs = [0, 5, 10],
difference = True, file_name = "Zbieganie_odleglosci.png")

prop44.plot_alphas_to_two(num_of_steps = 10000, h = 0.1,
func=exp_minus_squared, lim = laplacian_exp_minus_squared, points_indexs = [0, 5, 10],
difference = False, file_name = "Zbieganie_wartosci.png")