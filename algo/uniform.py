from typing import List

from algo.bai import UnimodalBaiAlgorithm


class UniformSampling(UnimodalBaiAlgorithm):
    NAME = "Uniform"

    def pull_arm(self) -> List[int]:
        return [self.t % self.n_arms]
