import numpy as np

import sys


def update_mean_online(mean_support: np.ndarray, mean_hat: np.ndarray, new_val: np.ndarray):
    return (1 / (1 + mean_support)) * (mean_support * mean_hat + new_val)


def conf_interval(std, num_runs, quantile=1.96):
    return quantile * std / (num_runs ** 0.5)


def inverse_bisection(f, y, x_min, x_max, tolerance, max_iterations=10000):
    if f(x_min) > y or f(x_max) < y:
        print("Inverse does not exist within the given range.")
        raise RuntimeError("Errors in inverse computation. Intervals ill-defined.")

    iterations = 0
    while iterations < max_iterations:
        x_mid = (x_min + x_max) / 2
        f_mid = f(x_mid)

        if abs(f_mid - y) < tolerance:
            return x_mid

        if f_mid < y:
            x_min = x_mid
        else:
            x_max = x_mid

        iterations += 1

    print("Maximum iterations reached. No convergence.")
    raise RuntimeError("Errors in inverse computation. Convergence failed.")


def compute_x_a(y, mu_1, mu_a, kl_f, tol_inverse):
    g_a = get_g_a(mu_1, mu_a, kl_f)
    return inverse_bisection(g_a, y, x_min=0, x_max=sys.maxsize, tolerance=tol_inverse)


def get_g_a(mu_1, mu_2, kl_f):
    def f(x):
        a1 = 1 / (1 + x)
        a2 = x / (1 + x)

        t1 = a1 * kl_f(mu_1, a1 * mu_1 + a2 * mu_2)
        t2 = a2 * kl_f(mu_2, a1 * mu_1 + a2 * mu_2)
        return (1 + x) * (t1 + t2)

    return f


def generalized_jensen_shannon(alpha: float, mu_1, mu_2, kl_f):
    t1 = alpha * kl_f(mu_1, alpha * mu_1 + (1 - alpha) * mu_2)
    t2 = (1 - alpha) * kl_f(mu_2, alpha * mu_1 + (1 - alpha) * mu_2)
    return t1 + t2


def is_perfect_integer(num):
    return num.is_integer()


def is_unimodal(model: np.array):
    n_arms = model.size

    a_star = np.argmax(model)
    for j in range(a_star):
        if model[j] > model[j + 1]:
            return False
    for j in range(a_star, n_arms-1):
        if model[j] < model[j + 1]:
            return False

    return True
