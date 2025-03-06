from typing import List

import numpy as np

from algo.bai import BaiConfig, UnstructuredBaiAlgorithm, UnimodalBaiAlgorithm


def check_unimodal_constraint(model: np.array):
    a_star = np.argmax(model)
    for i in range(a_star):
        if model[i] > model[a_star]:
            return False
    for i in range(a_star, model.size):
        if model[i] < model[a_star]:
            return False
    return True


def compute_structured_value_arms(alpha, beta, n_arms):
    # For the first arm, we need a loop over n elements
    opt_arms = []

    opt_model_0 = np.zeros(n_arms)
    opt_model_0[0] = beta[0]
    opt_model_0[-1] = alpha[-1]
    violation_set = set()
    for j in range(1, n_arms - 1)[::-1]:
        opt_model_0[j] = max([alpha[j], opt_model_0[j + 1]])
        if opt_model_0[j] < alpha[j] or opt_model_0[j] > beta[j]:
            violation_set.add(j)

    if len(violation_set) > 0 or opt_model_0[0] < opt_model_0[1]:
        opt_arms.append(-np.inf)
    else:
        opt_arms.append(opt_model_0[0])

    # For the remaining ones, we use dynamic programming to find the solution efficiently
    curr_opt_model = opt_model_0
    for i in range(1, n_arms):
        # Modify curr opt model
        curr_opt_model[i] = beta[i]
        curr_opt_model[i - 1] = alpha[i - 1] if i == 1 else max([alpha[i - 1], curr_opt_model[i - 2]])

        # Check violation list
        violation_set.discard(i)
        if alpha[i - 1] <= curr_opt_model[i - 1] <= beta[i - 1]:
            violation_set.discard(i - 1)
        else:
            violation_set.add(i - 1)

        # Update opt_arm list
        ok = True
        if len(violation_set) > 0:
            ok = False
        else:
            if i < n_arms - 1:
                if curr_opt_model[i] < curr_opt_model[i - 1] or curr_opt_model[i] < curr_opt_model[i + 1]:
                    ok = False
            else:
                if curr_opt_model[i] < curr_opt_model[i - 1]:
                    ok = False
        if ok:
            opt_arms.append(curr_opt_model[i])
        else:
            opt_arms.append(-np.inf)

    return opt_arms


def compute_leader_uni_tt_fast(alpha, beta, n_arms, num_active_models):
    opt_arms = compute_structured_value_arms(alpha, beta, n_arms)
    if num_active_models is not None:
        num_active_models.append(int(np.count_nonzero(np.array(opt_arms) != -np.inf)))
    return np.argmax(np.array(opt_arms))


def compute_leader_uni_tt_ts_structured(alpha, beta, n_arms, mean_hat, arm_count, num_active_models):
    opt_arms = compute_structured_value_arms(alpha, beta, n_arms)
    if num_active_models is not None:
        num_active_models.append(int(np.count_nonzero(np.array(opt_arms) != -np.inf)))
    ok = False
    while not ok:
        theta = np.random.normal(mean_hat, scale=np.sqrt(1 / arm_count))
        bt = np.argmax(theta)
        if opt_arms[bt] is not -np.inf:
            ok = True

    return bt


def compute_leader_uni_tt(alpha, beta, n_arms, num_active_models):
    vals = []

    counter = 0

    for arm_idx in range(n_arms):
        val = None
        opt_model = np.zeros(n_arms)
        opt_model[arm_idx] = beta[arm_idx]

        if arm_idx != 0:
            opt_model[0] = alpha[0]
        if arm_idx != n_arms - 1:
            opt_model[-1] = alpha[-1]

        # Computing optimistic models
        for a in range(1, arm_idx):
            if opt_model[a - 1] > beta[a]:
                val = -np.inf
            if opt_model[a - 1] < alpha[a]:
                opt_model[a] = alpha[a]
            else:
                opt_model[a] = opt_model[a - 1]
            if opt_model[arm_idx - 1] > opt_model[arm_idx]:
                val = -np.inf

        for a in range(arm_idx + 1, n_arms - 1)[::-1]:
            if opt_model[a + 1] > beta[a]:
                val = -np.inf
            if opt_model[a + 1] < alpha[a]:
                opt_model[a] = alpha[a]
            else:
                opt_model[a] = opt_model[a + 1]

            if opt_model[arm_idx] < opt_model[arm_idx + 1]:
                val = -np.inf

        # Style one
        if val is None:
            counter += 1
            vals.append(opt_model[arm_idx])
        else:
            vals.append(val)

    if num_active_models is not None:
        num_active_models.append(counter)

    return np.argmax(np.array(vals))


def uni_tt_sampling(n_arms, arm_count, mean_hat, alpha, beta, kl_f, tracker, sampling_strategy, num_active_models):
    if sampling_strategy == "struct_ucb":
        bt = compute_leader_uni_tt_fast(alpha, beta, n_arms, num_active_models)
        if bt is None:
            bt = np.argmax(mean_hat)
    elif sampling_strategy == "ucb":
        bt = np.argmax(beta)
    elif sampling_strategy == "ts":
        theta = np.random.normal(mean_hat, scale=np.sqrt(1 / arm_count))
        bt = np.argmax(theta)
    elif sampling_strategy == "ts_resampling":
        ok = False
        while not ok:
            theta = np.random.normal(mean_hat, scale=np.sqrt(1 / arm_count))
            if check_unimodal_constraint(theta):
                bt = np.argmax(theta)
    elif sampling_strategy == "ts_struct":
        bt = compute_leader_uni_tt_ts_structured(alpha, beta, n_arms, mean_hat, arm_count)
    else:
        raise RuntimeError(f"Strategy {sampling_strategy} not supported for Unimodal sampling")

    # Find challenger
    ct = None
    best_val = None
    for ct_guess in range(n_arms):
        if ct_guess == bt + 1 or ct_guess == bt - 1:
            n_b = arm_count[bt]
            n_c = arm_count[ct_guess]

            mu_b_c = (n_b / (n_c + n_b)) * mean_hat[bt] + (n_c / (n_c + n_b)) * mean_hat[ct_guess]
            if mean_hat[bt] < mean_hat[ct_guess]:
                mu_b_c = -mu_b_c
            curr_val = n_b * kl_f(mean_hat[bt], mu_b_c) + n_c * kl_f(mean_hat[ct_guess], mu_b_c)

            if ct is None or curr_val < best_val:
                best_val = curr_val
                ct = ct_guess

    # Tracking procedure
    next_arm = tracker.get_next_sample(bt, ct, arm_count)

    return [next_arm]


def bai_tt_sampling(n_arms, arm_count, mean_hat, optimistic_vals, kl_f, tracker, sampling_strategy):
    if sampling_strategy == "ucb":
        bt = np.argmax(optimistic_vals)
    elif sampling_strategy == "ts":
        theta = np.random.normal(loc=mean_hat, scale=np.sqrt(1 / arm_count))
        bt = np.argmax(theta)
    else:
        raise RuntimeError(f"Strategy {sampling_strategy} not supported for BAI sampling")

    # Find challenger
    ct = None
    best_val = None
    for ct_guess in range(n_arms):
        if ct_guess != bt:
            n_b = arm_count[bt]
            n_c = arm_count[ct_guess]

            mu_b_c = (n_b / (n_c + n_b)) * mean_hat[bt] + (n_c / (n_c + n_b)) * mean_hat[ct_guess]
            if mean_hat[bt] < mean_hat[ct_guess]:
                mu_b_c = -mu_b_c
            curr_val = n_b * kl_f(mean_hat[bt], mu_b_c) + n_c * kl_f(mean_hat[ct_guess], mu_b_c)

            if ct is None or curr_val < best_val:
                best_val = curr_val
                ct = ct_guess

    # Tracking procedure
    next_arm = tracker.get_next_sample(bt, ct, arm_count)

    return [next_arm]


class TopTwoTracking:

    def __init__(self, fixed_design: bool, n_arms: int):
        self.fixed_design = fixed_design
        self.n_arms = n_arms

        self.T_matrix = np.zeros((n_arms, n_arms))
        self.cum_beta = np.zeros((n_arms, n_arms))
        self.counter_when_leader = np.zeros((n_arms, n_arms))  # Vector of counts conditioned on which arm was leader

    def get_next_sample(self, b_t: int, c_t: int, arm_count: np.array):
        # Update T matrix and cumulative beta vector
        self.T_matrix[b_t, c_t] += 1
        beta_t = 0.5
        if self.fixed_design is False:
            beta_t = arm_count[c_t] / (arm_count[b_t] + arm_count[c_t])
        self.cum_beta[b_t, c_t] += beta_t

        # Tracking
        curr_avg_beta = self.cum_beta[b_t, c_t] / self.T_matrix[b_t, c_t]
        if self.counter_when_leader[b_t, c_t] <= (1 - curr_avg_beta) * self.T_matrix[b_t, c_t]:
            next_arm = c_t
        else:
            next_arm = b_t

        # Last update
        self.counter_when_leader[b_t, next_arm] += 1

        return next_arm


class TopTwoSamplingBai(UnstructuredBaiAlgorithm):
    NAME = "TT_BAI"

    def __init__(self, cfg: BaiConfig):
        super(TopTwoSamplingBai, self).__init__(cfg)

        self.tracker = TopTwoTracking(fixed_design=self.cfg.use_fixed_design, n_arms=self.n_arms)

    def pull_arm(self) -> List[int]:
        if self.t < self.n_arms:
            return [self.t]

        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        # Find leader
        _, optimistic_vals = self.compute_confidence_intervals()
        return bai_tt_sampling(self.cfg.n_arms,
                               self.arm_count,
                               self.mean_hat,
                               optimistic_vals,
                               self.cfg.kl_f,
                               self.tracker,
                               self.cfg.tt_sampling_strategy)


class TopTwoSamplingBaiStoppingUnimodalSampling(UnstructuredBaiAlgorithm):
    NAME = "TT_BAI_STOP_UNI_SAMPLING"

    def __init__(self, cfg: BaiConfig):
        super(TopTwoSamplingBaiStoppingUnimodalSampling, self).__init__(cfg)

        self.tracker = TopTwoTracking(fixed_design=self.cfg.use_fixed_design, n_arms=self.n_arms)

    def pull_arm(self) -> List[int]:
        if self.t < self.n_arms:
            return [self.t]

        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        alpha, beta = self.compute_confidence_intervals()
        return uni_tt_sampling(self.cfg.n_arms,
                               self.arm_count,
                               self.mean_hat,
                               alpha,
                               beta,
                               self.cfg.kl_f,
                               self.tracker,
                               self.cfg.tt_sampling_strategy
                               )


class TopTwoSamplingBaiSamplingUnimodalStopping(UnimodalBaiAlgorithm):
    NAME = "TT_BAI_SAMPLING_UNI_STOPPING"

    def __init__(self, cfg: BaiConfig):
        super(TopTwoSamplingBaiSamplingUnimodalStopping, self).__init__(cfg)

        self.tracker = TopTwoTracking(fixed_design=self.cfg.use_fixed_design, n_arms=self.n_arms)

    def pull_arm(self) -> List[int]:
        if self.t < self.n_arms:
            return [self.t]

        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        # Find leader
        _, optimistic_vals = self.compute_confidence_intervals()
        return bai_tt_sampling(self.cfg.n_arms,
                               self.arm_count,
                               self.mean_hat,
                               optimistic_vals,
                               self.cfg.kl_f,
                               self.tracker,
                               self.cfg.tt_sampling_strategy)


class TopTwoSamplingUnimodal(UnimodalBaiAlgorithm):
    NAME = "TT_UNIMODAL"

    def __init__(self, cfg: BaiConfig):
        super(TopTwoSamplingUnimodal, self).__init__(cfg)

        self.tracker = TopTwoTracking(fixed_design=self.cfg.use_fixed_design, n_arms=self.n_arms)

        self.num_active_models = None
        if self.cfg.store_num_active_model and (self.cfg.tt_sampling_strategy in ['struct_ucb', 'struct_ts']):
            self.num_active_models = []

    def pull_arm(self) -> List[int]:
        if self.t < self.n_arms:
            return [self.t]

        if self.cfg.verbose is not None and self.cfg.verbose > 0 and self.t % self.cfg.verbose == 0:
            print(f"Run id {self.cfg.run_id} ---- iter {self.t}")

        alpha, beta = self.compute_confidence_intervals()

        return uni_tt_sampling(self.cfg.n_arms,
                               self.arm_count,
                               self.mean_hat,
                               alpha,
                               beta,
                               self.cfg.kl_f,
                               self.tracker,
                               self.cfg.tt_sampling_strategy,
                               self.num_active_models
                               )
