from typing import List

import numpy as np

from algo.bai import BaiConfig, UnimodalBaiAlgorithm
from utils.unimodal_lower_bounds import solve_max_min_problem, AltSolver


def compute_optimistic_model(ci_alpha: np.ndarray,
                             ci_beta: np.ndarray,
                             arm_idx: int,
                             n_arms: int) -> np.array:
    opt_model = np.zeros(n_arms)
    opt_model[arm_idx] = ci_beta[arm_idx]

    if arm_idx != 0:
        opt_model[0] = ci_alpha[0]
    if arm_idx != n_arms - 1:
        opt_model[-1] = ci_alpha[-1]

    # Computing optimistic models
    for a in range(1, arm_idx):
        if opt_model[a - 1] > ci_beta[a]:
            return None
        if opt_model[a - 1] < ci_alpha[a]:
            opt_model[a] = ci_alpha[a]
        else:
            opt_model[a] = opt_model[a - 1]

    if arm_idx - 1 >= 0:
        if opt_model[arm_idx - 1] > opt_model[arm_idx]:
            return None

    for a in range(arm_idx + 1, n_arms - 1)[::-1]:
        if opt_model[a + 1] > ci_beta[a]:
            return None
        if opt_model[a + 1] < ci_alpha[a]:
            opt_model[a] = ci_alpha[a]
        else:
            opt_model[a] = opt_model[a + 1]

    if arm_idx + 1 <= n_arms - 1:
        if opt_model[arm_idx] < opt_model[arm_idx + 1]:
            return None

    return opt_model


class UnimodalOptTaSStruct(UnimodalBaiAlgorithm):
    NAME = "O-TaS"

    def __init__(self, cfg: BaiConfig):
        super(UnimodalOptTaSStruct, self).__init__(cfg)
        self.cum_weights = np.zeros(self.n_arms)
        self.alt_solver = AltSolver(3, self.cfg.kl_f, self.cfg.variance_proxy)

        self.num_active_models = None
        if self.cfg.store_num_active_model:
            self.num_active_models = []

    def pull_arm(self) -> List[int]:
        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        # Initialization
        if self.t < self.n_arms:
            return [self.t]

        # Sampling rule: compute oracle weights (by solving the optimistic problem)
        wt = self.compute_wt()
        self.cum_weights += wt

        # Sampling rule: Cumulative tracking
        arm_to_pull = np.argmax(self.cum_weights - self.arm_count)
        return [arm_to_pull]

    def compute_wt(self) -> np.array:
        # For all possible optimal arms, determine the optimistic model
        opt_mu_dict = {}
        counter = 0
        alpha, beta = self.compute_confidence_intervals()
        for a in range(self.n_arms):
            opt_mu = compute_optimistic_model(alpha, beta, a, self.cfg.n_arms)
            if opt_mu is not None:
                opt_mu_dict[a] = opt_mu
                counter += 1

        if self.num_active_models is not None:
            self.num_active_models.append(counter)

        # Solve the optimization problem to find the oracle weights
        best_val, best_weights, best_model, best_idx = None, None, None, None
        best_model_solved_three_arms, best_model_solved_three_arms_val = None, None
        for a in range(self.n_arms):
            if a in opt_mu_dict and 0 < a < self.n_arms - 1 and best_model_solved_three_arms is not None:
                if best_model_solved_three_arms[1] >= opt_mu_dict[a][a] and best_model_solved_three_arms[0] <= \
                        opt_mu_dict[a][a - 1] and best_model_solved_three_arms[2] <= opt_mu_dict[a][a + 1]:
                    continue

            if a in opt_mu_dict:
                if a == 0:
                    curr_w = solve_max_min_problem(3,
                                                   opt_mu_dict[a][0:3],
                                                   self.cfg.kl_f,
                                                   self.cfg.tol_F,
                                                   self.cfg.tol_inverse)
                    curr_val = self.alt_solver.solve(curr_w, opt_mu_dict[a][0:3])[1]
                    new_w = np.zeros(self.n_arms)
                    new_w[0:3] = curr_w
                elif a == self.n_arms - 1:
                    curr_w = solve_max_min_problem(3,
                                                   opt_mu_dict[a][-3:],
                                                   self.cfg.kl_f,
                                                   self.cfg.tol_F,
                                                   self.cfg.tol_inverse)
                    curr_val = self.alt_solver.solve(curr_w, opt_mu_dict[a][-3:])[1]
                    new_w = np.zeros(self.n_arms)
                    new_w[-3:] = curr_w
                else:
                    curr_w = solve_max_min_problem(3,
                                                   opt_mu_dict[a][a - 1:a + 2],
                                                   self.cfg.kl_f,
                                                   self.cfg.tol_F,
                                                   self.cfg.tol_inverse)
                    curr_val = self.alt_solver.solve(curr_w, opt_mu_dict[a][a - 1:a + 2])[1]
                    new_w = np.zeros(self.n_arms)
                    new_w[a - 1:a + 2] = curr_w

                    if best_model_solved_three_arms is None or curr_val > best_model_solved_three_arms_val:
                        best_model_solved_three_arms = opt_mu_dict[a][a - 1:a + 2]
                        best_model_solved_three_arms_val = curr_val

                if best_val is None or curr_val > best_val:
                    best_val = curr_val
                    best_weights = new_w
                    best_model = opt_mu_dict[a]
                    best_idx = a

        return best_weights
