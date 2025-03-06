from typing import List

import cvxpy as cp
import numpy as np

from algo.bai import BaiConfig, UnimodalBaiAlgorithm
from utils.unimodal_lower_bounds import solve_max_min_problem


class UnimodalTaS(UnimodalBaiAlgorithm):
    NAME = "Unimodal-TaS"

    def __init__(self, cfg: BaiConfig):
        super(UnimodalTaS, self).__init__(cfg)
        self.cum_weights = np.zeros(self.n_arms)

        # optimization
        self.opt_eps = cp.Parameter()
        self.opt_weights = cp.Parameter(self.n_arms)
        self.opt_proj_w = cp.Variable(self.n_arms)
        self.constraints = [cp.sum(self.opt_proj_w) == 1]
        for a in range(self.n_arms):
            self.constraints.append(self.opt_proj_w[a] >= self.opt_eps)

        self.obj = cp.norm(self.opt_proj_w - self.opt_weights, "inf")

        self.prob = cp.Problem(cp.Minimize(self.obj), self.constraints)

    def pull_arm(self) -> List[int]:
        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        # Initialization
        if self.t < self.n_arms:
            return [self.t]

        return [self.tracking()]

    def tracking(self) -> int:
        # Compute optimal empirical weights
        model = self.project(self.mean_hat)
        if np.argmax(model) == 0:
            curr_w = solve_max_min_problem(3,
                                           model[0:3],
                                           self.cfg.kl_f,
                                           self.cfg.tol_F,
                                           self.cfg.tol_inverse)
            new_w = np.zeros(self.n_arms)
            new_w[0:3] = curr_w
        elif np.argmax(model) == self.n_arms - 1:
            curr_w = solve_max_min_problem(3,
                                           model[-3:],
                                           self.cfg.kl_f,
                                           self.cfg.tol_F,
                                           self.cfg.tol_inverse)
            new_w = np.zeros(self.n_arms)
            new_w[-3:] = curr_w
        else:
            curr_w = solve_max_min_problem(3,
                                           model[np.argmax(model) - 1:np.argmax(model) + 2],
                                           self.cfg.kl_f,
                                           self.cfg.tol_F,
                                           self.cfg.tol_inverse)
            new_w = np.zeros(self.n_arms)
            new_w[np.argmax(model) - 1:np.argmax(model) + 2] = curr_w

        # Project weights on the eps-simplex to force exploration
        new_w = self.project_weights(new_w)
        # Add new weights to the cumulative vector
        self.cum_weights += new_w

        return np.argmax(self.cum_weights - self.arm_count)

    def project_weights(self, weights: np.array) -> np.array:
        eps = 0.5 * (self.n_arms ** 2 + self.t) ** (-1 / 2)
        self.opt_eps.value = eps
        self.opt_weights.value = weights
        self.prob.solve()

        return np.array(self.opt_proj_w.value)
