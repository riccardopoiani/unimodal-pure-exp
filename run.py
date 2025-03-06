import argparse
import os
import pickle

import numpy as np
import yaml
from joblib import Parallel, delayed

from algo.factory import BAIFactory
from algo.fw import FW
from algo.opt_tas import UnimodalOptTaSStruct
from algo.top_two import TopTwoSamplingBaiStoppingUnimodalSampling, TopTwoSamplingUnimodal
from envs.bandit import UnimodalBanditModel
from utils.math import conf_interval
from utils.results import ResultSummary, ResultSummaryNew
from utils.run_utils import read_cfg, mkdir_if_not_exist, run

np.set_printoptions(suppress=True, precision=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo",
                        type=str,
                        required=True,
                        choices=list(BAIFactory.algo_map.keys()))
    parser.add_argument("--env-cfg", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results/")
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--n-runs", type=int, default=1)
    parser.add_argument("--tol-F", type=float, default=1e-2)
    parser.add_argument("--tol-inverse", type=float, default=1e-8)
    parser.add_argument("--use-projection", type=int, default=0)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--constant-lr", type=float, default=0)
    parser.add_argument("--use-fixed-design", type=int, default=0)
    parser.add_argument("--use-naive-z", type=int, default=0)
    parser.add_argument("--max-iter", type=int, default=-1)
    parser.add_argument("--tt-sampling-strategy", type=str, default=None)
    parser.add_argument("--fw-forced-exp-not-model", type=int, default=0)
    parser.add_argument("--store-num-active-model", type=int, default=0)

    # Reading common arguments and read environment configuration
    args = parser.parse_args()
    env_cfg = read_cfg(args.env_cfg)

    use_projection = False
    if args.use_projection > 0:
        if args.algo != "RegretLearner" and args.algo != "Gradient":
            raise RuntimeError("Projection is only supported for problems that work according to the unimodal "
                               "structure.")
        use_projection = True

    constant_lr = None
    if args.constant_lr > 0:
        if args.algo != "RegretLearner" and args.algo != "Gradient":
            raise RuntimeError("This parameter is only available for RegretLearner and Gradient.")
        constant_lr = args.constant_lr

    use_fixed_design = False
    if args.use_fixed_design > 0:
        if args.algo not in BAIFactory.top_two_names:
            raise RuntimeError("This parameter is only available for TopTwo methods")
        use_fixed_design = True

    use_naive_z = False
    if args.use_naive_z > 0:
        if args.algo not in BAIFactory.unimodal_algo_name:
            raise RuntimeError("This parameter is only available for unimodal algorithms")
        use_naive_z = True

    if args.max_iter > 0:
        assert args.algo == TopTwoSamplingBaiStoppingUnimodalSampling.NAME

    tt_sampling_strategy = args.tt_sampling_strategy
    if tt_sampling_strategy is not None:
        if args.algo not in BAIFactory.top_two_names:
            raise RuntimeError("This parameter is only available for TopTwo methods")

    fw_forced_exp_not_model = args.fw_forced_exp_not_model
    if fw_forced_exp_not_model > 0 and args.algo != FW.NAME:
        raise RuntimeError(f"Parameter not supported for {args.algo}")

    store_num_active_model = args.store_num_active_model
    if store_num_active_model and (args.algo != UnimodalOptTaSStruct.NAME and args.algo != TopTwoSamplingUnimodal.NAME):
        raise RuntimeError(f"Parameter not supported for {args.algo}")

    # Launch pure-exploration
    seeds = [np.random.randint(1000000) for _ in range(args.n_runs)]
    if args.n_jobs == 1:
        results = [run(run_id=id,
                       seed=seed,
                       env_cfg=env_cfg,
                       algo_name=args.algo,
                       delta=args.delta,
                       tol_F=args.tol_F,
                       tol_inverse=args.tol_inverse,
                       use_projection=use_projection,
                       verbose=args.verbosity,
                       constant_lr=constant_lr,
                       use_fixed_design=use_fixed_design,
                       use_naive_z=use_naive_z,
                       max_iter=args.max_iter,
                       tt_sampling_strategy=tt_sampling_strategy,
                       fw_forced_exp_not_model=fw_forced_exp_not_model,
                       store_num_active_model=store_num_active_model)
                   for id, seed in zip(range(args.n_runs), seeds)]
    else:
        results = Parallel(n_jobs=args.n_jobs, backend='loky')(
            delayed(run)(run_id=id,
                         seed=seed,
                         env_cfg=env_cfg,
                         algo_name=args.algo,
                         delta=args.delta,
                         tol_F=args.tol_F,
                         tol_inverse=args.tol_inverse,
                         use_projection=use_projection,
                         verbose=args.verbosity,
                         constant_lr=constant_lr,
                         use_fixed_design=use_fixed_design,
                         use_naive_z=use_naive_z,
                         max_iter=args.max_iter,
                         tt_sampling_strategy=tt_sampling_strategy,
                         fw_forced_exp_not_model=fw_forced_exp_not_model,
                         store_num_active_model=store_num_active_model
                         )
            for id, seed in zip(range(args.n_runs), seeds))

    # Dump results on file
    mkdir_if_not_exist(args.results_dir)
    with open(os.path.join(args.results_dir, "results.pkl"), "wb") as output:
        pickle.dump(results, output)

    if store_num_active_model:
        res_summary = ResultSummaryNew(results, UnimodalBanditModel(**env_cfg))
    else:
        res_summary = ResultSummary(results, UnimodalBanditModel(**env_cfg))

    # Dump results
    summary = {
        'env_cfg': env_cfg,
        'algo': args.algo,
        'delta': args.delta,
        'n_runs': args.n_runs,
        'tol_F': args.tol_F,
        'tol_inverse': args.tol_inverse,
        'use_projection': use_projection,
        'constant_lr': constant_lr,
        'use_naive_z': use_naive_z,
        'use_fixed_design': use_fixed_design,
        'max_iter': args.max_iter,
        'tt_sampling_strategy': tt_sampling_strategy,
        'fw_forced_exp_not_model': fw_forced_exp_not_model,
        'results':
            {
                'correctness': res_summary.best_arm_stats(),
                'sample_complexity':
                    {
                        'mean': res_summary.sample_complexity_stats()[0].item(),
                        'std': res_summary.sample_complexity_stats()[1].item(),
                        'ci': conf_interval(res_summary.sample_complexity_stats()[1], args.n_runs).item()
                    }
            }
    }
    print(summary['results'])

    with open(os.path.join(args.results_dir, "run_specs.yml"), 'w') as outfile:
        yaml.dump(summary, outfile, default_flow_style=False)
