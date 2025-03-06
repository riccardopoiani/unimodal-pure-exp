from algo.bai import BaiAlgorithm
from envs.bandit import UnimodalBandit


def learn(algo: BaiAlgorithm, env: UnimodalBandit) -> (int, int):
    """
    :param algo: algorithm to be used
    :param env: env in which agent will run
    :return: (best arm, cost complexity)
    """
    while not algo.stopping_condition():
        curr_arms = algo.pull_arm()
        rewards = []
        for a in curr_arms:
            r = env.step(a)
            rewards.append(r)
        algo.update(curr_arms, rewards)

    return algo.recommendation()
