from math import floor
from typing import List

import numpy as np
import cvxpy as cp

from algo.bai import BaiConfig, UnimodalBaiAlgorithm
from utils.math import is_perfect_integer, is_unimodal
from utils.unimodal_lower_bounds import AltSolver


class MaxMinGameSolverItem:

    def __init__(self, k, j):
        # Opt vars
        self.u = cp.Variable(1)
        self.z = cp.Variable(k)

        # param
        self.M = cp.Parameter((k, j))

        # game constraints
        self.constraints = []
        for i in range(j):
            self.constraints.append(self.z @ self.M[:, i] >= self.u)

        # Simplex constraints
        for i in range(k):
            self.constraints.append(self.z[i] >= 0)
            self.constraints.append(self.z[i] <= 1)
        self.constraints.append(cp.sum(self.z) == 1)

        self.obj = self.u
        self.prob = cp.Problem(cp.Maximize(self.obj), self.constraints)

    def solve(self, M) -> np.ndarray:
        self.M.value = M

        try:
            self.prob.solve()
        except:
            try:
                self.prob.solve(solver=cp.CVXOPT)
            except:
                raise RuntimeError

        return self.z.value


class MaxMinGameSolver:

    def __init__(self, n_arms):
        self.problem_list = [MaxMinGameSolverItem(n_arms, i + 1) for i in range(n_arms)]

    def solve(self, M):
        j = M.shape[1]
        return self.problem_list[j - 1].solve(M)


class FW(UnimodalBaiAlgorithm):
    NAME = "FW"

    def __init__(self, cfg: BaiConfig):
        super(FW, self).__init__(cfg)
        self.x = np.ones(self.cfg.n_arms) / self.cfg.n_arms
        self.w = np.zeros(self.cfg.n_arms)

        self.basis_vector_list = []
        for a in range(self.cfg.n_arms):
            vec = np.eye(1, self.cfg.n_arms, a)
            self.basis_vector_list.append(vec)

        self.alt_solver = AltSolver(self.n_arms, self.cfg.kl_f, self.cfg.variance_proxy)
        self.max_min_solver = MaxMinGameSolver(self.cfg.n_arms)

    def pull_arm(self) -> List[int]:
        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        if self.t < self.n_arms:
            return [self.t]

        # Compute zeta
        if self.is_forced_exp():
            zeta = np.ones(self.cfg.n_arms) / self.cfg.n_arms
        else:
            zeta = self.fw_step(self.x)

        # Compute empirical proportions
        omega = self.arm_count / self.t

        # Update xt
        self.x = ((self.t - 1) / self.t) * self.x + (1 / self.t) * zeta

        return [np.argmax(self.x / omega)]

    def fw_step(self, x_vec) -> np.ndarray:
        omega = self.arm_count / self.t

        # Compute f-values and all the gradients
        f_dict, grad_dict = {}, {}
        f_list = []
        for a in range(self.n_arms):
            if a != np.argmax(self.mean_hat):
                f_a, alt_a = self.alt_solver.solvers[a].solve(omega.copy(), self.mean_hat.copy())
                f_dict[a] = f_a
                grad_dict[a] = np.array([(self.cfg.kl_f(self.mean_hat[a], alt_a[a])) for a in range(self.n_arms)])
                f_list.append(f_a)

        F = min(f_list)

        # Find good indexes
        good_idxs = []
        for a in range(self.n_arms):
            if a in f_dict and f_dict[a] < F + self.get_r():
                good_idxs.append(a)

        # Compute M matrix
        M = np.zeros((self.cfg.n_arms, len(good_idxs)))
        for a in range(self.n_arms):
            for i in range(len(good_idxs)):
                M[a, i] = np.dot(self.basis_vector_list[a] - x_vec, grad_dict[good_idxs[i]])
        return self.max_min_solver.solve(M)

    def is_forced_exp(self):
        # Check if perfect integer
        if is_perfect_integer((floor(self.t / self.cfg.n_arms)) ** 0.5):
            return True

        # Check if unimodal
        if self.cfg.fw_forced_exp_not_model:
            if is_unimodal(self.mean_hat):
                return False
            return True

        return False

    def get_r(self):
        return self.t ** (-9.0 / 10) / self.cfg.n_arms
