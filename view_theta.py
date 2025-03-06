import argparse
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from envs.bandit import UnimodalBanditModel
from utils.math import conf_interval
from utils.results import ResultSummaryNew
from utils.run_utils import read_cfg, mkdir_if_not_exist

algo_labels = [
    'ttuni',
    'opt-tas-struct'
]

algo_labels_map = {
    'ttuni': "UniTT",
    'opt-tas-struct': "O-TaS",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str)
    parser.add_argument("--store", type=int, default=0)
    parser.add_argument("--dir-name", type=str, default=None)
    parser.add_argument("--filter", type=int, default=1)

    args = parser.parse_args()

    # Specifying setting
    folder = f"results/log/{args.exp}"
    path_list = [f"results/logfix/{args.exp}/{name}/results.pkl" for name in algo_labels]
    info_file_path = f"results/logfix/{args.exp}/{algo_labels[0]}/run_specs.yml"

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
    res_sum_dict = {}
    for key in data.keys():
        res_sum_dict[key] = ResultSummaryNew(res_list=data[key], bandit_model=env_model)

    # Computing dataframe
    x_data_by_algo = {}
    m_by_algo = {}
    for i, algo in enumerate(algo_labels):
        # Read num model list
        res_list = res_sum_dict[algo].res_list
        res_list_num_models = [r.num_active_models for r in res_list]

        # Find max iter
        m = 0
        for l in res_list_num_models:
            if len(l) > m:
                m = len(l)
        m_by_algo[algo] = m

        # Create x_data
        x_data_mean = np.zeros(m)
        x_data_count = np.zeros(m)
        x_data_std = np.zeros(m)

        for t in range(m):
            data_at_t = np.array([r[t] for r in res_list_num_models if t < len(r)])
            if t == 0:
                print(data_at_t.mean())
            x_data_mean[t] = data_at_t.mean()
            x_data_std[t] = data_at_t.std()
            x_data_count[t] = data_at_t.size

        x_data_by_algo[algo] = (x_data_mean, x_data_std, x_data_count)

    # Plot results
    color_list = ['blue', 'green']
    line_style_list = ['-', '--']
    for i, algo in enumerate(algo_labels):
        filter_arr = np.array([i for i in range(m_by_algo[algo]) if i % args.filter == 0])
        x_axis = np.arange(m_by_algo[algo])
        plt.plot(x_axis[filter_arr],
                 x_data_by_algo[algo][0][filter_arr],
                 label=algo_labels_map[algo],
                 color=color_list[i],
                 linestyle=line_style_list[i],
                 linewidth=2)
        ci = conf_interval(x_data_by_algo[algo][1], x_data_by_algo[algo][2])
        plt.fill_between(x_axis[filter_arr],
                         x_data_by_algo[algo][0][filter_arr] - ci[filter_arr],
                         x_data_by_algo[algo][0][filter_arr] + ci[filter_arr],
                         color=color_list[i],
                         alpha=0.2)

    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.xlabel('Iter. Number')
    plt.ylabel('Number of Active Arms')
    plt.legend()

    if args.store > 0:
        if args.dir_name is None:
            mkdir_if_not_exist(f"results/plots/{args.exp}/")
            plt.savefig(f"results/plots/{args.exp}/plot.png", dpi=250, bbox_inches='tight')
        else:
            mkdir_if_not_exist(f"results/plots/{args.dir_name}/")
            plt.savefig(f"results/plots/{args.dir_name}/plot.png", dpi=250, bbox_inches='tight')
    else:
        plt.show()
