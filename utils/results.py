from typing import List

import numpy as np

from envs.bandit import UnimodalBanditModel


class ResultItem:
    __slots__ = ["best_arm",
                 "sample_complexity",
                 "num_pulls_per_arm"]

    def __init__(self,
                 best_arm,
                 sample_complexity,
                 num_pulls_per_arm):
        self.best_arm = best_arm
        self.sample_complexity = sample_complexity
        self.num_pulls_per_arm = num_pulls_per_arm


class ResultItemNew:
    __slots__ = ["best_arm",
                 "sample_complexity",
                 "num_pulls_per_arm",
                 "num_active_models"]

    def __init__(self,
                 best_arm,
                 sample_complexity,
                 num_pulls_per_arm,
                 num_active_models):
        self.best_arm = best_arm
        self.sample_complexity = sample_complexity
        self.num_pulls_per_arm = num_pulls_per_arm
        self.num_active_models = num_active_models


class ResultSummaryNew:
    def __init__(self, res_list: List[ResultItemNew], bandit_model: UnimodalBanditModel):
        self.model = bandit_model
        self.res_list = res_list
        self.num_res = len(self.res_list)

    @property
    def num_run(self):
        return self.num_res

    def return_sample_complexity_array(self) -> np.array:
        res = [res.sample_complexity for res in self.res_list]
        return np.array(res)

    def best_arm_stats(self):
        """
        :return: (percentage of right identifications)
        """
        true_best_arm = self.model.get_best_arm()
        count = 0
        for res in self.res_list:
            if res.best_arm == true_best_arm:
                count += 1
        return count / self.num_res * 100

    def get_arm_pull_data(self):
        data = np.zeros((self.num_res, self.model.n_arms))
        for i, res in enumerate(self.res_list):
            data[i] = res.num_pulls_per_arm

        return data

    def sample_complexity_stats(self):
        """
        :return: (mean, std, all_vals) of cost complexity required to identify the best arm
        """
        all_vals = np.array([res.sample_complexity for res in self.res_list])
        return all_vals.mean(), all_vals.std(), all_vals


class ResultSummary:

    def __init__(self, res_list: List[ResultItem], bandit_model: UnimodalBanditModel):
        self.model = bandit_model
        self.res_list = res_list
        self.num_res = len(self.res_list)

    @property
    def num_run(self):
        return self.num_res

    def return_sample_complexity_array(self) -> np.array:
        res = [res.sample_complexity for res in self.res_list]
        return np.array(res)

    def best_arm_stats(self):
        """
        :return: (percentage of right identifications)
        """
        true_best_arm = self.model.get_best_arm()
        count = 0
        for res in self.res_list:
            if res.best_arm == true_best_arm:
                count += 1
        return count / self.num_res * 100

    def get_arm_pull_data(self):
        data = np.zeros((self.num_res, self.model.n_arms))
        for i, res in enumerate(self.res_list):
            data[i] = res.num_pulls_per_arm

        return data

    def sample_complexity_stats(self):
        """
        :return: (mean, std, all_vals) of cost complexity required to identify the best arm
        """
        all_vals = np.array([res.sample_complexity for res in self.res_list])
        return all_vals.mean(), all_vals.std(), all_vals
