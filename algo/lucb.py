from typing import List
import numpy as np

from algo.bai import BaiAlgorithm, BaiConfig


class LUCB(BaiAlgorithm):
    NAME = "LUCB"

    def __init__(self, cfg: BaiConfig):
        super(LUCB, self).__init__(cfg)

    def get_ht(self) -> int:
        return np.argmax(self.mean_hat)

    def get_lt(self) -> int:
        mean_copy = self.mean_hat.copy()
        M = np.max(self.mean_hat)
        mean_copy[np.where(mean_copy == M)] = -np.inf
        _, beta = self.compute_confidence_intervals()
        return np.argmax(mean_copy + np.sqrt(2 * (np.log(self.n_arms / self.delta) + np.log(np.log(self.t))) / self.arm_count))
        # return np.argmax(mean_copy + beta)

    def stopping_condition(self) -> bool:
        if self.t < self.n_arms:
            return False

        lt = self.get_lt()
        ht = self.get_ht()

        ci = np.sqrt(2 * (np.log(self.n_arms / self.delta) + np.log(np.log(self.t))) / self.arm_count)

        if self.mean_hat[lt] + ci[lt] < self.mean_hat[ht] - ci[ht]:
            return True

        return False

    def pull_arm(self) -> List[int]:
        if self.t < self.n_arms:
            return [self.t]
        return [self.get_ht(), self.get_lt()]

    def recommendation(self) -> int:
        return self.get_ht()
