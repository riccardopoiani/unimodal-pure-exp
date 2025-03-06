import numpy as np

from utils.distribution import kl_gaussian
from utils.math import is_unimodal
from utils.unimodal_lower_bounds import solve_max_min_problem, AltSolver

np.set_printoptions(suppress=True)

"""
if __name__ == "__main__":
    kl_f = kl_gaussian(1)
    solver = AltSolver(3, kl_f, 1)

    arm = 0
    mu_1 = np.array([0.67, 1.26, 0.09])
    mu_2 = np.array([0.37, 1.72, 0.09])
    mu = [mu_1, mu_2]
    for m in mu:
        res = solve_max_min_problem(3, m, kl_f, tol_F=1e-2, tol_inverse=1e-8)
        alt = solver.solve(res, m)[0]
        print(alt)
# [0.49409814 0.49423388 0.01166798]
# [2.8536448  2.8737228  2.78024455]
"""
if __name__ == "__main__":
    a = np.array([-5, 11, 4, 3, 2, 2])
