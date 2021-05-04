from Proposition_4_4 import Proposition44
from utils_func_essential import exp_minus_squared

prop44 = Proposition44()
prop44.plot_alphas_to_zero(num_of_steps = 10000, h=0.1,
func=exp_minus_squared, points_indexs=[0, 5, 10], difference = True)