from abc import abstractmethod
from typing import List

import cvxpy as cp
import numpy as np

from algo.bai import BaiConfig, UnstructuredBaiAlgorithm
from utils.unstructured_lower_bounds import solve_max_min_problem_unstructured


class TrackAndStop(UnstructuredBaiAlgorithm):
    """
    Track And Stop. "Optimal Best Arm Identification with Fixed Confidence", Garivier and Kaufmann, 2016.
    Optimistic Track and Stop. "Non-Asymptotic Pure Exploration by Solving Games", Degenne et al., 2019.
    """

    def __init__(self, cfg: BaiConfig):
        super(TrackAndStop, self).__init__(cfg)

    def pull_arm(self) -> List[int]:
        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        # Initialization
        if self.t < self.n_arms:
            return [self.t]

        return [self.tracking()]

    @abstractmethod
    def tracking(self) -> int:
        raise NotImplementedError


class CTracking(TrackAndStop):
    NAME = "TaS-C-Unstruct"

    def __init__(self, cfg: BaiConfig):
        super(CTracking, self).__init__(cfg)

        self.cum_weights = np.zeros(self.n_arms)

    def tracking(self) -> int:
        # Compute optimal empirical weights
        w = solve_max_min_problem_unstructured(self.n_arms,
                                               self.mean_hat.copy(),
                                               self.cfg.kl_f,
                                               self.cfg.tol_F,
                                               self.cfg.tol_inverse)

        # Project weights on the eps-simplex to force exploration
        w = self.projection(w)

        # Add new weights to the cumulative vector
        self.cum_weights += w

        return np.argmax(self.cum_weights - self.arm_count)

    def projection(self, weights: np.array) -> np.array:
        eps = 0.5 * (self.n_arms ** 2 + self.t) ** (-1 / 2)

        new_w = cp.Variable(self.n_arms)
        constraints = [cp.sum(new_w) == 1]
        for a in range(self.n_arms):
            constraints.append(new_w[a] >= eps)

        obj = cp.norm(new_w - weights, "inf")

        prob = cp.Problem(cp.Minimize(obj), constraints)
        prob.solve()

        return np.array(new_w.value)


class DTracking(TrackAndStop):
    NAME = "TaS-D-Unstruct"

    def __init__(self, cfg: BaiConfig):
        super(DTracking, self).__init__(cfg)

    def tracking(self) -> int:
        # Forced exploration of arms that are under-sampled
        for a in range(self.n_arms):
            if self.arm_count[a] < np.sqrt(self.t) - (self.n_arms / 2):
                return a

        w = solve_max_min_problem_unstructured(self.n_arms,
                                               self.mean_hat.copy(),
                                               self.cfg.kl_f,
                                               self.cfg.tol_F,
                                               self.cfg.tol_inverse)

        # Direct tracking
        return np.argmax(self.t * w - self.arm_count)
