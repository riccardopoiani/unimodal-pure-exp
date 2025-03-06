from abc import ABC, abstractmethod
from typing import List

import numpy as np

from utils.math import update_mean_online


class BaiConfig:
    __slots__ = ['n_arms',
                 'delta',
                 'variance_proxy',
                 'kl_f',
                 'dist_type',
                 'tol_F',
                 'tol_inverse',
                 'run_id',
                 'verbose',
                 'constant_lr',
                 'use_projection',
                 'use_fixed_design',
                 'use_naive_z',
                 'max_iter',
                 'tt_sampling_strategy',
                 'fw_forced_exp_not_model',
                 'store_num_active_model']

    def __init__(self,
                 n_arms,
                 delta,
                 variance_proxy,
                 kl_f,
                 dist_type,
                 tol_F,
                 tol_inverse,
                 run_id,
                 verbose,
                 constant_lr,
                 use_projection,
                 use_fixed_design,
                 use_naive_z,
                 max_iter,
                 tt_sampling_strategy,
                 fw_forced_exp_not_model,
                 store_num_active_model
                 ):
        self.n_arms = n_arms
        self.delta = delta
        self.variance_proxy = variance_proxy
        self.kl_f = kl_f
        self.dist_type = dist_type
        self.tol_F = tol_F
        self.tol_inverse = tol_inverse
        self.run_id = run_id
        self.verbose = verbose
        self.constant_lr = constant_lr
        self.use_projection = use_projection
        self.use_fixed_design = use_fixed_design
        self.use_naive_z = use_naive_z
        self.max_iter = max_iter
        self.tt_sampling_strategy = tt_sampling_strategy
        self.fw_forced_exp_not_model = fw_forced_exp_not_model,
        self.store_num_active_model = store_num_active_model


class BaiAlgorithm(ABC):

    def __init__(self, cfg: BaiConfig):
        self.n_arms = cfg.n_arms
        self.delta = cfg.delta
        self.variance_proxy = cfg.variance_proxy

        self.mean_hat = np.zeros(self.n_arms)
        self.arm_count = np.zeros(self.n_arms)

        self.cfg = cfg

        self.t = 0

    @abstractmethod
    def stopping_condition(self) -> bool:
        """
        :return: True if the algorithm needs to stop, False otherwise
        """
        raise NotImplementedError

    @abstractmethod
    def pull_arm(self) -> List[int]:
        """
        Returns which arm to pull

        :return: list of arm_idx to be pulled
        """
        raise NotImplementedError

    @abstractmethod
    def recommendation(self) -> int:
        """
        :return: which is the best arm
        """
        raise NotImplementedError

    def update(self, arm_idxes: List[int], rewards: List):
        # Update mean rewards and arm count
        for a, r in zip(arm_idxes, rewards):
            self.mean_hat[a] = update_mean_online(
                self.arm_count[a],
                self.mean_hat[a],
                r
            )
            self.arm_count[a] += 1

        self.t += len(rewards)

    def get_arm_count(self) -> List[int]:
        """
        :return: number of pulls for each arm
        """
        return self.arm_count.tolist()

    def get_sample_complexity(self):
        return self.arm_count.sum()

    def compute_threshold(self) -> float:
        return np.log(self.n_arms / self.delta) + np.log(np.log(self.t) + 1)

    def compute_confidence_intervals(self) -> (np.array, np.array):
        if self.cfg.dist_type != "gaussian":
            raise RuntimeError("Only Gaussian distributions are supported.")
        alpha = self.mean_hat - np.sqrt((4 * self.cfg.variance_proxy * np.log(self.t)) / self.arm_count)
        beta = self.mean_hat + np.sqrt((4 * self.cfg.variance_proxy * np.log(self.t)) / self.arm_count)

        # bar_w_arg = 2.88 * np.log(self.t) + 2 * np.log(2 + 1.2 * np.log(self.t)) + 2
        # g_t = bar_w_arg + np.log(bar_w_arg)

        # alpha = self.mean_hat - np.sqrt(g_t / self.arm_count)
        # beta = self.mean_hat + np.sqrt(g_t / self.arm_count)

        # alpha = self.mean_hat - np.sqrt((np.log(self.t) + np.log(np.log(self.t))) / self.arm_count)
        # beta = self.mean_hat + np.sqrt((np.log(self.t) + np.log(np.log(self.t))) / self.arm_count)

        return alpha, beta


class UnstructuredBaiAlgorithm(BaiAlgorithm, ABC):

    def stopping_condition(self) -> bool:
        if 0 < self.cfg.max_iter <= self.t:
            return True
        if self.t < self.n_arms:
            return False

        # Compute stopping rule
        z_list = []
        a = np.argmax(self.mean_hat)
        n_a = self.arm_count[a]
        curr_z = None
        for b in range(self.n_arms):
            if b != a:
                n_b = self.arm_count[b]
                mu_a_b = (n_a / (n_a + n_b)) * self.mean_hat[a] + (n_b / (n_a + n_b)) * self.mean_hat[b]
                z_a_b = n_a * self.cfg.kl_f(self.mean_hat[a], mu_a_b) + n_b * self.cfg.kl_f(self.mean_hat[b],
                                                                                            mu_a_b)
                if self.mean_hat[a] < self.mean_hat[b]:
                    z_a_b = -z_a_b

                if curr_z is None or curr_z > z_a_b:
                    curr_z = z_a_b
        z_list.append(curr_z)

        if max(z_list) > self.compute_threshold():
            return True
        return False

    def recommendation(self) -> int:
        return np.argmax(self.mean_hat)


class UnimodalBaiAlgorithm(BaiAlgorithm, ABC):

    def project(self, model):
        model = model.copy()
        opt_idx = np.argmax(model)

        proj = np.zeros(self.cfg.n_arms)
        proj[opt_idx] = model[opt_idx]

        for a in range(opt_idx)[::-1]:
            if model[a] > proj[a + 1]:
                proj[a] = proj[a + 1]
            else:
                proj[a] = model[a]
        for a in range(opt_idx + 1, self.cfg.n_arms):
            if model[a] > proj[a - 1]:
                proj[a] = proj[a - 1]
            else:
                proj[a] = model[a]

        return proj

    def compute_z_list(self):
        if self.cfg.use_naive_z:
            # Compute stopping rule using only the empirical best arm
            z_list = []
            a = np.argmax(self.mean_hat)
            n_a = self.arm_count[a]
            curr_z = None
            for b in range(self.n_arms):
                if b == a - 1 or b == a + 1:
                    n_b = self.arm_count[b]
                    mu_a_b = (n_a / (n_a + n_b)) * self.mean_hat[a] + (n_b / (n_a + n_b)) * self.mean_hat[b]
                    z_a_b = n_a * self.cfg.kl_f(self.mean_hat[a], mu_a_b) + n_b * self.cfg.kl_f(self.mean_hat[b],
                                                                                                mu_a_b)
                    if self.mean_hat[a] < self.mean_hat[b]:
                        z_a_b = -z_a_b

                    if curr_z is None or curr_z > z_a_b:
                        curr_z = z_a_b
            z_list.append(curr_z)
        else:
            # Compute stopping rule as in the proposed algorithm
            z_list = []
            for a in range(self.n_arms):
                n_a = self.arm_count[a]
                curr_z = None

                for b in range(self.n_arms):
                    if b == a - 1 or b == a + 1:
                        n_b = self.arm_count[b]
                        mu_a_b = (n_a / (n_a + n_b)) * self.mean_hat[a] + (n_b / (n_a + n_b)) * self.mean_hat[b]
                        z_a_b = n_a * self.cfg.kl_f(self.mean_hat[a], mu_a_b) + n_b * self.cfg.kl_f(self.mean_hat[b],
                                                                                                    mu_a_b)
                        if self.mean_hat[a] < self.mean_hat[b]:
                            z_a_b = -z_a_b

                        if curr_z is None or curr_z > z_a_b:
                            curr_z = z_a_b
                z_list.append(curr_z)
        return z_list

    def stopping_condition(self) -> bool:
        if 0 < self.cfg.max_iter <= self.t:
            return True
        if self.t < self.n_arms:
            return False

        z_list = self.compute_z_list()

        if max(z_list) > self.compute_threshold():
            return True
        return False

    def recommendation(self) -> int:
        if self.cfg.use_naive_z:
            return np.argmax(self.mean_hat)
        z_list = self.compute_z_list()
        return np.argmax(z_list)
