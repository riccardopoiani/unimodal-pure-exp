import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from envs.bandit import UnimodalBanditModel
from utils.results import ResultSummary
from utils.run_utils import read_cfg, mkdir_if_not_exist

main_algo_list = [
    'Unimodal-TaS',
    'uni-tt',
    #'uni-tt-ts',
    'opt-tas-struct',
    'tt-bai-ucb',
    "TaS-C-Unstruct",
    'Uniform',
    "Gradient",
    'fwexp1',
    "RegretLearner",
    # 'uni-tt-ts-struct',
    # 'tt-bai-ts',
    # 'fwexp0',
]

stop_ablation_algo_list = [
    "uni-tt-naive-z",  # Previous version of the stopping rule
    "uni-tt",  # Our algorithm
    "bai-stop-uni-sampling-struct"  # Bai stopping, sampling: ours
]

ablation_uni_tt = [
    'uni-tt',  # Sampling: ours, Stopping: Ours
    'uni-tt-ucb',  # Sampling: no structured CI, Stopping: Ours
    # 'bai-stop-uni-sampling-struct',  # Sampling: ours, Stopping: BAI
    'tt-bai-sampling-uni-stopping',  # Sampling: UCB, Stopping: Ours
    'tt-bai-ucb'
]

algo_labels_map = {
    "Gradient": "LMA",
    "RegretLearner": "DKM",
    "TaS-C-Unstruct": "TaS",
    'Unimodal-TaS': "U-TaS",
    'tt-bai-ucb': "TTUCB",
    'uni-tt': "UniTT",
    'uni-tt-ucb': "UniTT-UCB",
    'uni-tt-ts': "UTT-TS",
    'uni-tt-ts-struct': "UTT-TS-S",
    'Uniform': 'Unif.',
    'bai-stop-uni-sampling-struct': "UTT-TS-S-BaiStop",
    'bai-sampling-uni-stopping-ucb': "TT-UCB-UNIStop",
    'uni-tt-naive-z': "UTT-UCB-S-Z",
    'opt-tas-struct': "O-TaS",
    'fwexp1': 'FW',
    'fwexp0': 'FW0',
    'tt-bai-sampling-uni-stopping': 'TT-UniStop',
    'uni-tt-ucb': 'UniTT-NO-S'
}

fast_algos = ["tt-bai-ts",
              "tt-bai-ucb",
              "uni-tt",
              "uni-tt-ucb",
              "uni-tt-ts",
              "uni-tt-ts-struct",
              "Uniform",
              "Unimodal-TaS",
              "opt-tas-struct"]


def filter_algo(algo_list, use_fast_algos, exclude_uniform, exclude_u_tas):
    if exclude_uniform:
        algo_list.remove("Uniform")
    if exclude_u_tas:
        algo_list.remove("Unimodal-TaS")

    if use_fast_algos:
        res = []
        for a in algo_list:
            if a in fast_algos:
                res.append(a)
        return res
    return algo_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--main-algo", type=int, default=0)
    parser.add_argument("--ablation-uni-tt", type=int, default=0)
    parser.add_argument("--ablation-stopping", type=int, default=0)
    parser.add_argument("--fast-algos", type=int, default=0)
    parser.add_argument("--exclude-uniform", type=int, default=0)
    parser.add_argument("--exclude-u-tas", type=int, default=0)
    parser.add_argument("--num-pulls", type=int, default=0)
    parser.add_argument("--store", type=int, default=0)
    parser.add_argument("--name", type=str, default=None)

    args = parser.parse_args()

    # Algo labels
    if args.main_algo:
        algo_labels = filter_algo(main_algo_list, args.fast_algos, args.exclude_uniform, args.exclude_u_tas)
    elif args.ablation_uni_tt:
        algo_labels = filter_algo(ablation_uni_tt, args.fast_algos, args.exclude_uniform, args.exclude_u_tas)
    elif args.ablation_stopping:
        algo_labels = filter_algo(stop_ablation_algo_list, args.fast_algos, args.exclude_uniform, args.exclude_u_tas)
    else:
        raise RuntimeError("Setting not specified.")
    print(algo_labels)

    # Specifying setting
    folder = f"results/{args.exp}"
    path_list = [f"results/{args.exp}/{name}/results.pkl" for name in algo_labels]
    info_file_path = f"results/{args.exp}/{algo_labels[0]}/run_specs.yml"

    # Read result file
    data = {}
    for i, algo_name in enumerate(algo_labels):
        with open(path_list[i], "rb") as f:
            data[algo_name] = pickle.load(f)
    info_file = read_cfg(info_file_path)

    # Generic + build unimodal bandit model of the underlying experiment
    n_runs = info_file['n_runs']
    n_algos = len(algo_labels)
    env_cfg = info_file['env_cfg']
    env_model = UnimodalBanditModel(**env_cfg)

    # Convert data to result summary
    result_summary_by_algo = {}
    for key in data.keys():
        result_summary_by_algo[key] = ResultSummary(res_list=data[key], bandit_model=env_model)

    # Algorithm correctness
    for i, algo in enumerate(algo_labels):
        print(f"Algorithm {algo}: correctness {result_summary_by_algo[algo].best_arm_stats()}")
    print()

    # Visualize box plot and print performance
    x_data = np.zeros((n_algos, n_runs))
    for i, algo in enumerate(algo_labels):
        x_data[i] = result_summary_by_algo[algo].return_sample_complexity_array()

    algo_labels_mapped = [algo_labels_map[l] for l in algo_labels]
    x_data = pd.DataFrame(np.transpose(x_data), columns=algo_labels_mapped)

    sns.set(style="whitegrid")
    # plt.figure(fig-size=(5, 4))
    plt.figure(facecolor=(1, 1, 1))
    plt.ylabel("Empirical Sample Complexity", fontsize=14)
    boxplot = sns.boxplot(data=x_data, showfliers=False, linewidth=2.5)
    plt.grid(True, which='both', axis='y', color='gray', linestyle='--', linewidth=0.7)
    boxplot.set_facecolor('whitesmoke')

    # Customize the spines (borders)
    sns.despine(offset=10, trim=True)

    if args.store > 0:
        if args.name is None:
            mkdir_if_not_exist(f"results/plots/new/{args.exp}/")
            plt.savefig(f"results/plots/new/{args.exp}/plot.png", dpi=250, bbox_inches='tight')
        else:
            mkdir_if_not_exist(f"results/plots/new/{args.name}/")
            plt.savefig(f"results/plots/new/{args.name}/plot.png", dpi=250, bbox_inches='tight')
    else:
        plt.show()

    # Visualize number of pulls
    if args.num_pulls > 0:
        data_opt = result_summary_by_algo['uni-tt'].get_arm_pull_data().mean(axis=0)
        data_tas = result_summary_by_algo['opt-tas-struct'].get_arm_pull_data().mean(axis=0)
        data_tas_un = result_summary_by_algo['tt-bai-ucb'].get_arm_pull_data().mean(axis=0)
        print(data_opt)

        x_axis = np.arange(data_opt.shape[0])
        plt.plot(x_axis, data_opt, marker='x', markersize=10, label="FW0")
        plt.plot(x_axis, data_tas, marker='x', markersize=10, label="FW1")
        plt.plot(x_axis, data_tas_un, marker='x', markersize=10, label="TaS-C-Unstruct")
        plt.legend()
        plt.show()
