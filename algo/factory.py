from algo.bai import BaiConfig, BaiAlgorithm
from algo.fw import FW
from algo.gradient import Gradient
from algo.lucb import LUCB
from algo.opt_tas import UnimodalOptTaSStruct
from algo.regret import RegretLearner
from algo.tas import DTracking, CTracking
from algo.top_two import TopTwoSamplingBai, TopTwoSamplingUnimodal, TopTwoSamplingBaiSamplingUnimodalStopping, \
    TopTwoSamplingBaiStoppingUnimodalSampling
from algo.uniform import UniformSampling
from algo.unimodal_tas import UnimodalTaS


class BAIFactory:
    algo_map = {DTracking.NAME: DTracking,
                CTracking.NAME: CTracking,
                RegretLearner.NAME: RegretLearner,
                TopTwoSamplingBai.NAME: TopTwoSamplingBai,
                TopTwoSamplingUnimodal.NAME: TopTwoSamplingUnimodal,
                Gradient.NAME: Gradient,
                UnimodalTaS.NAME: UnimodalTaS,
                UniformSampling.NAME: UniformSampling,
                TopTwoSamplingBaiSamplingUnimodalStopping.NAME: TopTwoSamplingBaiSamplingUnimodalStopping,
                TopTwoSamplingBaiStoppingUnimodalSampling.NAME: TopTwoSamplingBaiStoppingUnimodalSampling,
                UnimodalOptTaSStruct.NAME: UnimodalOptTaSStruct,
                FW.NAME: FW,
                LUCB.NAME: LUCB
                }

    top_two_names = [TopTwoSamplingBai.NAME,
                     TopTwoSamplingUnimodal.NAME,
                     TopTwoSamplingBaiSamplingUnimodalStopping.NAME,
                     TopTwoSamplingBaiStoppingUnimodalSampling.NAME
                     ]

    unimodal_algo_name = [
        RegretLearner.NAME,
        TopTwoSamplingUnimodal.NAME,
        Gradient.NAME,
        UnimodalTaS.NAME
    ]

    @staticmethod
    def get_algo(algo_name: str,
                 bai_cfg: BaiConfig
                 ) -> BaiAlgorithm:
        assert algo_name in list(BAIFactory.algo_map.keys()), f"Algorithm {algo_name} is not implemented."

        return BAIFactory.algo_map[algo_name](bai_cfg)