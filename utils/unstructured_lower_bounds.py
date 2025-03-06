import numpy as np

from utils.math import compute_x_a


def solve_max_min_problem_unstructured(n_arms: int,
                                       mu_vec: np.array,
                                       kl_f,
                                       tol_F,
                                       tol_inverse,
                                       num_iter=0,
                                       eps=1e-2,
                                       re_iter=False) -> np.array:
    if num_iter > 5:
        raise RuntimeError

    mu_star = mu_vec.max()

    # First detect if there are multiple arms that are equal
    if not re_iter:
        opt_arm_idx = np.argmax(mu_vec)
        mu_vec[opt_arm_idx] += eps

    # Solve optimization problem
    mu_set = [mu_vec[a] for a in range(n_arms) if a != np.argmax(mu_vec)]

    y_min = 0
    y_max = kl_f(mu_star, np.max(np.array(mu_set)))
    y = (y_max + y_min) / 2

    # Search for y^* (by bisection)
    i = 0
    while abs(evaluate_f_y_unstruct(y, mu_vec, kl_f, tol_inverse) - 1) > tol_F:
        if evaluate_f_y_unstruct(y, mu_vec, kl_f, tol_inverse) > 1:
            y_max = y
        else:
            y_min = y

        y = (y_max + y_min) / 2
        i += 1
        if i > 2000:
            return solve_max_min_problem_unstructured(n_arms,
                                                      mu_vec,
                                                      kl_f,
                                                      tol_F,
                                                      tol_inverse * 0.1,
                                                      eps=0.05,
                                                      num_iter=num_iter + 1,
                                                      re_iter=True)

    return compute_w_from_y_unstruct(mu_vec, y, kl_f, tol_inverse)


def evaluate_f_y_unstruct(y, mu_vec, kl_f, tol_inverse):
    f = 0
    mu_star = np.max(mu_vec)

    for m in mu_vec:
        if m != mu_vec.max():
            x_a = compute_x_a(y, mu_star, m, kl_f, tol_inverse)

            num = kl_f(mu_star, (mu_star + x_a * m) / (1 + x_a))
            den = kl_f(m, (mu_star + x_a * m) / (1 + x_a))

            f += num / den

    return f


def compute_w_from_y_unstruct(mu_vec, y, kl_f, tol_inverse):
    # Compute x
    x_vals = np.zeros(mu_vec.size)
    for a in range(mu_vec.size):
        if a == np.argmax(mu_vec):
            x_vals[a] = 1
        else:
            x_vals[a] = compute_x_a(y, mu_vec[np.argmax(mu_vec)], mu_vec[a], kl_f, tol_inverse)

    # Compute w^*
    weights = x_vals / x_vals.sum()

    return weights
