import argparse
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.stats import bootstrap

from envs.bandit import UnimodalBanditModel
from utils.results import ResultSummary
from utils.run_utils import read_cfg

algo_labels = [#'Unimodal-TaS',
               'uni-tt-ucb',
               'uni-tt',
               'uni-tt-ts',
               'uni-tt-ts-struct',
               'tt-bai-ts',
               'tt-bai-ucb',
               ]

marker_list = ['o', 's', '^', 'D', 'v', 'x', '*']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, choices=['rnd', 'flat'])
    args = parser.parse_args()

    folder_list = []
    if args.exp == 'rnd':
        folder_list = [
            'results/rnd10',
            'results/rnd100',
            'results/rnd500',
            # 'results/rnd1000'
        ]
        ticks = [10, 100, 500]
    elif args.exp == 'flat':
        folder_list = [
            'results/flat11',
            'results/flat101',
            'results/flat501'
        ]
        ticks = [11, 101, 501]
    else:
        raise RuntimeError("Unspecified setting.")

    algo_results = {}
    for algo in algo_labels:
        algo_results[algo] = []

    # Read results
    for folder in folder_list:
        path_list = [f"{folder}/{name}/results.pkl" for name in algo_labels]
        info_file_path = f"{folder}/{algo_labels[0]}/run_specs.yml"

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
        for i, algo in enumerate(algo_labels):
            curr_data = result_summary_by_algo[algo].return_sample_complexity_array()
            mean = curr_data.mean()
            ci = bootstrap(curr_data.reshape((1, n_runs)), np.std, confidence_level=0.95)
            low = ci.confidence_interval.low
            high = ci.confidence_interval.high
            algo_results[algo].append((mean, low, high))

    # Visualize results
    num_exps = len(folder_list)
    ticks = np.array(ticks)
    for i, algo in enumerate(algo_labels):
        algo_mean = np.array([elem[0] for elem in algo_results[algo]])
        algo_low = np.array([elem[1] for elem in algo_results[algo]])
        algo_high = np.array([elem[2] for elem in algo_results[algo]])

        plt.plot(ticks, algo_mean, label=algo, marker=marker_list[i])
        plt.fill_between(ticks, algo_mean - algo_low, algo_mean + algo_high, alpha=0.2)

    plt.legend()
    plt.show()