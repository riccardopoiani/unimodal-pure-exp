from typing import List

import numpy as np

from algo.bai import BaiConfig, UnimodalBaiAlgorithm
from utils.regret_minimizer import AdaHedge
from utils.unimodal_lower_bounds import AltSolver


class RegretLearner(UnimodalBaiAlgorithm):
    NAME = "DKM"

    def __init__(self, cfg: BaiConfig):
        super(RegretLearner, self).__init__(cfg)
        self.ada_hedge = AdaHedge(self.cfg.n_arms, constant_lr=self.cfg.constant_lr)
        self.cum_w = np.zeros(self.cfg.n_arms)

        self.alt_solver = AltSolver(self.n_arms, self.cfg.kl_f, self.cfg.variance_proxy)

    def pull_arm(self) -> List[int]:
        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        # Uniform exploration
        if self.t < self.n_arms:
            return [self.t]

        # Compute new weights from the regret minimizer
        if self.t == self.n_arms:
            curr_w = np.ones(self.cfg.n_arms) / self.cfg.n_arms
        else:
            curr_w = self.ada_hedge.get_action()
        self.feed_loss_regret_min(curr_w)
        self.cum_w += curr_w

        # Tracking
        return [np.argmin(self.arm_count / self.cum_w)]

    def feed_loss_regret_min(self, curr_w):
        model = self.mean_hat.copy()
        if self.cfg.use_projection:
            model = self.project(self.mean_hat)

        # first term
        t1 = np.log(self.t - 1) / self.arm_count

        alt_problem = self.alt_solver.solve(curr_w, model)[0]
        alpha, beta = self.compute_confidence_intervals()
        lb = self.cfg.kl_f(alpha, alt_problem)
        ub = self.cfg.kl_f(beta, alt_problem)
        t2 = np.maximum(lb, ub)

        loss = -np.maximum(t1, t2)

        self.ada_hedge.feed(loss)
