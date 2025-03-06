from typing import List

import numpy as np

from utils.distribution import DistributionFactory


class UnimodalBanditModel:
    __slots__ = ['arms',
                 'dist_type',
                 'other_fixed_dist_param',
                 'n_arms']

    def __init__(self,
                 n_arms: int,
                 arms: List[List[float]],
                 dist_type: str,
                 other_fixed_dist_param):
        assert len(arms) == n_arms

        self.n_arms = n_arms
        self.dist_type = dist_type
        self.other_fixed_dist_param = other_fixed_dist_param
        self.arms = []
        for a in range(n_arms):
            self.arms.append(DistributionFactory.get_dist(dist_type, arms[a], other_fixed_dist_param))

        self._verify()

    def get_var_proxy(self):
        return self.arms[0].get_var_proxy()

    def _verify(self):
        # Assert that optimal arm is unique
        best_arm_idx = self.get_best_arm()
        max_val = self.arms[best_arm_idx].get_mean()
        for a in range(self.n_arms):
            if a != best_arm_idx:
                assert self.arms[a].get_mean() < max_val

        # Assert that the structure is unimodal
        for a in range(0, best_arm_idx):
            assert self.arms[a].get_mean() <= self.arms[a+1].get_mean()

        for a in range(best_arm_idx, self.n_arms - 1):
            assert self.arms[a].get_mean() >= self.arms[a+1].get_mean()

    def get_best_arm(self) -> int:
        """
        :return: idx of the best arm
        """
        vals = np.array([self.arms[arm].get_mean() for arm in range(self.n_arms)])
        return np.argmax(vals)


class UnimodalBandit:

    def __init__(self, model: UnimodalBanditModel):
        self.model = model

    def step(self, arm_idx: int) -> float:
        """
        :param arm_idx: index  of the arms to be pulled
        :return: reward
        """
        assert self.model.n_arms > arm_idx >= 0, f"Attempting to play arm {arm_idx}"

        return self.model.arms[arm_idx].sample()
