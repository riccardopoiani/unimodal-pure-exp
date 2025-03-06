import os
import random
import yaml
import numpy as np

from algo.bai import BaiConfig
from algo.factory import BAIFactory
from algo.learn import learn
from envs.bandit import UnimodalBanditModel, UnimodalBandit
from utils.distribution import DistributionFactory
from utils.results import ResultItem, ResultItemNew


def mkdir_if_not_exist(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def fix_seed(seed_val):
    if seed_val is not None:
        os.environ["PYTHONHASHSEED"] = str(seed_val)

        random.seed(seed_val)
        np.random.seed(seed_val)


def read_cfg(env_cfg_path: str):
    with open(env_cfg_path, "r") as f:
        env_cfg = yaml.load(f, Loader=yaml.FullLoader)

    return env_cfg


def build_bai_cfg(bandit_model: UnimodalBanditModel,
                  delta,
                  variance_proxy,
                  tol_F,
                  tol_inverse,
                  use_projection,
                  run_id,
                  verbose,
                  constant_lr,
                  use_fixed_design,
                  use_naive_z,
                  max_iter,
                  tt_sampling_strategy,
                  fw_forced_exp_not_model,
                  store_num_active_model):
    d = {'n_arms': bandit_model.n_arms,
         'delta': delta,
         'variance_proxy': variance_proxy,
         'kl_f': DistributionFactory.get_kl_f(bandit_model.dist_type, bandit_model.other_fixed_dist_param),
         'dist_type': bandit_model.dist_type,
         'tol_F': tol_F,
         'tol_inverse': tol_inverse,
         'use_projection': use_projection,
         'run_id': run_id,
         'verbose': verbose,
         'constant_lr': constant_lr,
         'use_fixed_design': use_fixed_design,
         'use_naive_z': use_naive_z,
         'max_iter': max_iter,
         'tt_sampling_strategy': tt_sampling_strategy,
         'fw_forced_exp_not_model': fw_forced_exp_not_model,
         'store_num_active_model': store_num_active_model
         }
    return BaiConfig(**d)


def run(run_id,
        seed,
        env_cfg,
        algo_name,
        delta,
        tol_F,
        tol_inverse,
        use_projection,
        verbose,
        constant_lr,
        use_fixed_design,
        use_naive_z,
        max_iter,
        tt_sampling_strategy,
        fw_forced_exp_not_model,
        store_num_active_model):
    print(f"Run {run_id} started.")

    # Fix seed
    fix_seed(seed)

    # Instantiate env and agents
    env_model = UnimodalBanditModel(**env_cfg)
    env = UnimodalBandit(env_model)

    bai_cfg = build_bai_cfg(env_model,
                            delta,
                            env_model.get_var_proxy(),
                            tol_F,
                            tol_inverse,
                            use_projection,
                            run_id,
                            verbose,
                            constant_lr,
                            use_fixed_design,
                            use_naive_z,
                            max_iter,
                            tt_sampling_strategy,
                            fw_forced_exp_not_model,
                            store_num_active_model)
    algo = BAIFactory.get_algo(algo_name, bai_cfg)

    # Learn
    best_arm = learn(algo, env)

    print(f"Run {run_id} completed. Stop at {algo.arm_count.sum()}")

    # Prepare results
    if store_num_active_model:
        return ResultItemNew(best_arm,
                             algo.get_sample_complexity(),
                             algo.get_arm_count(),
                             algo.num_active_models)
    return ResultItem(best_arm,
                      algo.get_sample_complexity(),
                      algo.get_arm_count())
