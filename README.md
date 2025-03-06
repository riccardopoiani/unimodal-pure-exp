# Code for the paper "Best-Arm Identification in Unimodal Bandits" (AISTATS '25)

This repository is based on `python 3.8`.
The necessary dependencies can be found in the `requirements.txt` file.

### Algorithms
The codebase provide an implementation of the following algorithms.
- Track and Stop for unstructured bandits. We provide both the C-Tracking and the D-Tracking version. These algorithms 
  are referred to as `TaS-C-Unstruct` and `TaS-D-Unstruct` respectively.
- Asymptotically Optimal algorithm for Unimodal Bandits, i.e., `DKM`, `LMA`, and `FW`.
- Unimodal-Track and Stop. This algorithm is referred to as `Unimodal-TaS` and has been presented in our paper.
- Optimistic Track and Stop for Unimodal bandits with structure-aware confidence intervals `O-TaS`.
- Top-Two Sampling algorithms. Concerning these class of algorithms, we provide an implementation of the following methods.
  - `TTUCB` for unstructured BAI. This is referred to as `TT_BAI`. To run this algo, one need to append `--tt-sampling-strategy ucb` to the run script (see below).
  - `UniTT` that we proposed in our algorithm. This is referred to as `TT_UNIMODAL`. To run this algo, one will need to append `--tt-sampling-strategy struct-ucb` to the run script.
  - `UniTT-BAIStop` which uses the sampling rule of `UniTT` but stops with a BAI GLR. This algo is referred to as `TT_BAI_STOP_UNI_SAMPLING`. To run this algo, one need to append `--tt-sampling-strategy struct-ucb` to the run script. 
  - `UniTT-NO-S`, which uses `UniTT` but without the structured confidence intervals. This is referred to as `TT_UNIMODAL`. To run this algo, one will need to append `--tt-sampling-strategy ucb`.
  - `TT-UniStop`, which uses `TTUCB` but with the unimodal stopping rule. This algo is referred to as `TT_BAI_SAMPLING_UNI_STOPPING`. To run this algo, one will need to append `--tt-sampling-strategy ucb`.
- Uniform sampling with the GLR that we proposed in our paper. This algorithm is referred to as `Uniform`. 

### Running the experiments
In the `configs` directory, one can find the different bandit models that has been used in the paper.
Specifically `flat_K.yml` for `K in {11, 101}` are configuration files for flat instances with `11` and `101` respectively.
Similarly, `rndK.yml` for `K in {10, 100, 500, 1000}` are configuration files for random instances with `10`, `100`, 
`500` and `1000` arms respectively. These models have been generated using the `generate_rnd_instance.py` script.
To generate additional random instances with `N` arms one can run the following command `python generate_rnd_instance.py --n-arms N --file-name rndN`.

To run an algorithm with a specific configuration file, it is sufficient to use the `run.py` script. A typical usage is reported as follows:
`python run.py --algo TaS-C-Unstruct --env-cfg configs/rnd10.yml --delta 0.1 --n-jobs 8 --n-runs 16 --result-dir results/`.
This will launch the Track and Stop algorithm with C-Tracking on the unimodal bandit specified in `configs/rnd10.yml`.
The experiment is run with a maximum risk parameter delta of `0.1`. Moreover, `--n-runs 16` specify that the experiment 
will be repeated `16` times, and the execution is parallelized among `8` runs (i.e., `--n-jobs 8`). 
Empirical results will be stored in the folder `results` specified right after `--result-dir`.
The results file contains a `ResultSummary` object (`utils/results.py`) which contains all the data from the experiments.

Finally, to reproduce the ablation on `\widetilde{\Theta}_t`, add `--store-num-active-model 1` when running the experiment 
with `O-TaS` and `TT_UNIMODAL`. In this case, the results is a `ResultSummaryNew` object.




