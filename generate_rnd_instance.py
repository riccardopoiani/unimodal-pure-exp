import argparse
import os
import yaml
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-arms", type=int, required=True)
    parser.add_argument("--variance-proxy", type=float, default=1.0)
    parser.add_argument("--max-gap", type=float, default=0.1)
    parser.add_argument("--min-gap-best-arm", type=float, default=0.2)
    parser.add_argument("--max-gap-best-arm", type=float, default=0.3)
    parser.add_argument("--best-arm-idx", type=int, default=None)
    parser.add_argument("--file-name", type=str, required=True)

    args = parser.parse_args()

    # Parameters
    n_arms = args.n_arms
    best_arm_idx = args.best_arm_idx
    min_gap_best_arm = args.min_gap_best_arm
    max_gap_best_arm = args.max_gap_best_arm
    max_gap = args.max_gap

    # Create random unimodal instance
    if best_arm_idx is None:
        best_arm_idx = np.random.randint(low=0, high=n_arms)
    arm_list = np.zeros(n_arms)
    arm_list[best_arm_idx] = 1.0

    i = best_arm_idx - 1
    while i >= 0:
        # Arm creation
        if i == best_arm_idx - 1:
            gap = np.random.uniform(min_gap_best_arm, max_gap_best_arm)
        else:
            gap = np.random.uniform(0, max_gap)

        arm_list[i] = arm_list[i+1] - gap

        # Counter
        i -= 1

    i = best_arm_idx + 1
    while i < n_arms:
        # Arm creation
        if i == best_arm_idx + 1:
            gap = np.random.uniform(min_gap_best_arm, max_gap_best_arm)
        else:
            gap = np.random.uniform(0, max_gap)

        arm_list[i] = arm_list[i - 1] - gap

        # Counter
        i += 1

    d = {
        'arms': arm_list.tolist(),
        'dist_type': 'gaussian',
        'n_arms': n_arms,
        'other_fixed_dist_param': args.variance_proxy
    }

    with open(os.path.join(f"configs/{args.file_name}.yml"), 'w') as outfile:
        yaml.dump(d, outfile, default_flow_style=False)
