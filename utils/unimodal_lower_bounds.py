import cvxpy as cp
import numpy as np

from utils.math import compute_x_a


class AltSolverPerIndex:
    def __init__(self, n_arms: int, opt_idx: int, kl_f, var_proxy):
        self.theta = cp.Variable(n_arms)
        self.constraints = []
        self.var_proxy = var_proxy

        self.mu = cp.Parameter(n_arms)
        self.w = cp.Parameter(n_arms, nonneg=True)
        self.w_times_mu = cp.Parameter(n_arms)
        self.w_times_mu_squared = cp.Parameter(n_arms)
        self.min_value = cp.Parameter()
        self.max_value = cp.Parameter()

        # Add unimodal constraints
        for a in range(n_arms):
            self.constraints.append(self.theta[a] >= self.min_value)
            self.constraints.append(self.theta[a] <= self.max_value)
            if a < opt_idx:
                self.constraints.append(self.theta[a] <= self.theta[a + 1])
            if opt_idx <= a < n_arms - 1:
                self.constraints.append(self.theta[a] >= self.theta[a + 1])

        # self.obj = cp.sum(cp.multiply(self.w, kl_f(self.mu, self.theta)))
        t1 = cp.sum(self.w_times_mu_squared)
        t2 = cp.sum(cp.multiply(self.w, cp.square(self.theta)))
        t3 = -2 * cp.sum(cp.multiply(self.w_times_mu, self.theta))

        self.obj = 1/(2 ** self.var_proxy) * (t1 + t2 + t3)
        self.prob = cp.Problem(cp.Minimize(self.obj), self.constraints)
        if not self.prob.is_dpp:
            raise RuntimeError

    def solve(self, w, mu):
        self.mu.value = mu
        self.w.value = w
        self.w_times_mu.value = w * mu
        self.w_times_mu_squared.value = w * (mu ** 2)
        self.min_value.value = mu.min()
        self.max_value.value = mu.max()

        try:
            sol = self.prob.solve()
        except:
            try:
                sol = self.prob.solve(solver=cp.CVXOPT)
            except:
                raise RuntimeError

        return sol, self.theta.value


class AltSolver:

    def __init__(self, n_arms, kl_f, variance_proxy):
        self.n_arms = n_arms
        self.solvers = [AltSolverPerIndex(n_arms, i, kl_f, variance_proxy) for i in range(n_arms)]

    def solve(self, w, mu):
        best_arm_idx = np.argmax(mu)

        # First try to solve using eta
        f_list = []
        best_val, best_alt, best_idx = None, None, None
        for a in range(self.n_arms):
            if a != best_arm_idx:
                f_a, alt = self.solvers[a].solve(w, mu)
                f_list.append(f_a)
                if best_val is None or f_a < best_val:
                    best_val = f_a
                    best_alt = alt
                    best_idx = a

        return best_alt, best_val, best_idx


def solve_max_min_problem(n_arms,
                          mu_vec: np.array,
                          kl_f,
                          tol_F,
                          tol_inverse,
                          num_iter=0,
                          eps=1e-01,
                          re_iter=False) -> np.array:
    if num_iter >= 5:
        raise RuntimeError

    if not re_iter:
        opt_arm_idx = np.argmax(mu_vec)
        mu_vec[opt_arm_idx] += eps

    # Compute y
    mu_star = mu_vec.max()
    opt_arm_idx = np.argmax(mu_vec)

    e_set = []
    mu_set = []
    if opt_arm_idx > 0:
        e_set.append(opt_arm_idx - 1)
        mu_set.append(mu_vec[opt_arm_idx - 1])
    if opt_arm_idx < n_arms - 1:
        e_set.append(opt_arm_idx + 1)
        mu_set.append(mu_vec[opt_arm_idx + 1])

    y_min = 0
    y_max = kl_f(mu_star, np.max(np.array(mu_set)))
    y = (y_max + y_min) / 2

    # Search for y^* (by bisection)
    i = 0
    while abs(evaluate_f_y(y, mu_star, mu_set, kl_f, tol_inverse) - 1) > tol_F:
        if evaluate_f_y(y, mu_star, mu_set, kl_f, tol_inverse) > 1:
            y_max = y
        else:
            y_min = y

        y = (y_max + y_min) / 2
        i += 1
        if i > 1000:
            return solve_max_min_problem(n_arms, mu_vec, kl_f, tol_F, tol_inverse * 0.1, num_iter + 1, re_iter=False)

    opt_arm_idx = np.argmax(mu_vec)
    return compute_w_from_y(n_arms, mu_vec, opt_arm_idx, y, kl_f, tol_inverse)


def evaluate_f_y(y, mu_star, e_means, kl_f, tol_inverse):
    f = 0
    for m in e_means:
        x_a = compute_x_a(y, mu_star, m, kl_f, tol_inverse)

        num = kl_f(mu_star, (mu_star + x_a * m) / (1 + x_a))
        den = kl_f(m, (mu_star + x_a * m) / (1 + x_a))

        f += num / den

    return f


def compute_w_from_y(n_arms, mu_vec: np.array, opt_arm_idx: int, y: float, kl_f, tol_inverse):
    # Build E(\bm{\mu})
    e_set = [opt_arm_idx]
    if opt_arm_idx > 0:
        e_set.append(opt_arm_idx - 1)
    if opt_arm_idx < n_arms - 1:
        e_set.append(opt_arm_idx + 1)

    # Compute x
    x_vals = np.zeros(n_arms)
    for idx in e_set:
        if idx == opt_arm_idx:
            x_vals[idx] = 1
        else:
            x_vals[idx] = compute_x_a(y, mu_vec[opt_arm_idx], mu_vec[idx], kl_f, tol_inverse)

    # Compute w^*
    weights = x_vals / x_vals.sum()

    return weights
