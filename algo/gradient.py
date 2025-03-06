from typing import List

import numpy as np

from algo.bai import BaiConfig, UnimodalBaiAlgorithm
from utils.regret_minimizer import GradientAscent
from utils.unimodal_lower_bounds import AltSolver


class Gradient(UnimodalBaiAlgorithm):
    NAME = "LMA"

    def __init__(self, cfg: BaiConfig):
        super(Gradient, self).__init__(cfg)
        self.cum_w_forced = np.zeros(self.n_arms)
        self.uniform_vector = np.ones(self.n_arms) / self.n_arms
        self.gradient_ascent = GradientAscent(self.n_arms, constant_lr=self.cfg.constant_lr)
        self.curr_w_tilde = np.ones(self.n_arms) / self.n_arms

        self.alt_solver = AltSolver(self.n_arms, self.cfg.kl_f, self.cfg.variance_proxy)

    def pull_arm(self) -> List[int]:
        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        if self.t < self.n_arms:
            return [self.t]

        # Arm pull
        gradient = self.compute_gradient()
        self.gradient_ascent.feed(gradient)
        self.curr_w_tilde = self.gradient_ascent.get_action()

        # Add forced exploration
        w_prime = (1 - self.get_gamma()) * self.curr_w_tilde + self.get_gamma() * self.uniform_vector
        self.cum_w_forced += w_prime

        # Tracking
        a = np.argmax(self.cum_w_forced - self.arm_count)
        return [a]

    def compute_gradient(self):
        model = self.mean_hat
        if self.cfg.use_projection:
            model = self.project(self.mean_hat)

        # Find alternative problem
        alt_problem = self.alt_solver.solve(self.curr_w_tilde, model)[0]

        # Compute subgradient
        grad = [(self.cfg.kl_f(model[a], alt_problem[a])) for a in range(self.n_arms)]

        return np.array(grad)

    def get_gamma(self):
        return 1 / (4 * np.sqrt(self.t))
