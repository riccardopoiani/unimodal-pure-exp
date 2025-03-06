import numpy as np


class AdaHedge:

    def __init__(self, dim, constant_lr=None):
        self.dim = dim
        self.L = np.zeros(dim)
        self.delta = 0.01
        self.constant_lr = constant_lr

    def get_action(self) -> np.array:
        eta = self.get_lr()
        u = np.exp(-eta * (self.L - np.min(self.L)))
        return u / u.sum()

    def feed(self, loss: np.array):
        u = self.get_action()
        eta = self.get_lr()
        M_pre = self.M(self.L, eta)
        self.L += loss
        M_post = self.M(self.L, eta)
        m = M_post - M_pre
        self.delta += np.dot(u, loss) - m

    def M(self, L, eta):
        if eta == 0:
            return np.sum(L) / self.dim

        return np.min(L) - (1 / eta) * np.log(np.sum(np.exp(-eta * (L - np.min(L)))) / self.dim)

    def get_lr(self):
        if self.constant_lr is not None:
            return self.constant_lr
        return np.log(self.dim) / self.delta


class GradientAscent:
    """
    Exponential weights algorithm
    """

    def __init__(self, dim: int, constant_lr=None):
        self.dim = dim
        self.t = 0
        self.g_t = np.zeros(dim, dtype=np.float128)
        self.constant_lr = constant_lr

    def get_action(self) -> np.array:
        lr = self.get_lr()
        return np.exp(lr * (self.g_t - np.min(self.g_t))) / np.sum(np.exp(lr * (self.g_t - np.min(self.g_t))))

    def feed(self, gradient):
        self.t += 1
        self.g_t += gradient

    def get_lr(self):
        if self.constant_lr is not None:
            return self.constant_lr
        return np.sqrt(1 / (self.t + 1))
