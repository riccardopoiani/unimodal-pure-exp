from abc import ABC, abstractmethod

import numpy as np


def kl_gaussian(std):
    def kl(q1, q2):
        return ((q1 - q2) ** 2) / (2 * std ** 2)

    return kl


class Distribution(ABC):

    def __init__(self):
        pass

    @staticmethod
    @abstractmethod
    def get_name() -> str:
        raise NotImplementedError

    @abstractmethod
    def sample(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_mean(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def kl(self, q1) -> float:
        raise NotImplementedError

    @abstractmethod
    def get_var_proxy(self) -> float:
        raise NotImplementedError


class GaussianDist(Distribution):

    def __init__(self, mean: float, std: float):
        super(GaussianDist, self).__init__()
        self.mean = mean
        self.std = std

    @staticmethod
    def get_name() -> str:
        return "gaussian"

    def get_mean(self) -> float:
        return self.mean

    def sample(self):
        return np.random.normal(loc=self.mean, scale=self.std)

    def kl(self, q1) -> float:
        return np.log(q1.std / self.std) + ((self.std ** 2 + (self.mean - q1.mean) ** 2) / (2 * q1.std ** 2)) - 0.5

    def get_var_proxy(self) -> float:
        return self.std ** 2


class DistributionFactory:
    name_to_dist = {GaussianDist.get_name(): GaussianDist}

    @staticmethod
    def get_dist(name: str, param, other_dist_param):
        return DistributionFactory.name_to_dist[name](param, other_dist_param)

    @staticmethod
    def get_kl_f(name: str, other_dist_param):
        return kl_gaussian(other_dist_param)