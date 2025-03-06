cores = '12-112'
n_runs = 1000
n_jobs = 100
cfg_list = ['rnd1000.yml']
cfg_names = {
    'rnd10.yml': 'rnd10',
    'rnd100.yml': 'rnd100',
    'rnd500.yml': 'rnd500',
    'rnd1000.yml': 'rnd1000',
    'flat_11.yml': 'flat11',
    'flat_101.yml': 'flat101',
    'flat_501.yml': 'flat501',
}
delta = 0.01
verbosity = 500

algo_names = ['Uniform',
              'TaS-C-Unstruct',
              'RegretLearner',
              'TT_BAI',
              'TT_UNIMODAL',
              'Gradient',
              'Unimodal-TaS',
              'TT_BAI_SAMPLING_UNI_STOPPING',
              'TT_BAI_STOP_UNI_SAMPLING'
              ]

additional_params = {
    'TT_UNIMODAL': [
        '--tt-sampling-strategy struct_ucb',
        '--tt-sampling-strategy ucb',
        '--tt-sampling-strategy ts',
        '--tt-sampling-strategy struct_ucb --use-naive-z 1',
        '--tt-sampling-strategy struct_ucb --use-fixed-design 1'
    ],
    'TT_BAI_STOP_UNI_SAMPLING': [
        '--tt-sampling-strategy struct_ucb --max-iter 50000',
        '--tt-sampling-strategy ucb --max-iter 50000',
        '--tt-sampling-strategy ts --max-iter 50000'
    ],
    'TT_BAI_SAMPLING_UNI_STOPPING': [
        '--tt-sampling-strategy ucb',
        '--tt-sampling-strategy ts'
    ],
    'TT_BAI': [
        '--tt-sampling-strategy ucb',
        '--tt-sampling-strategy ts'
    ]

}

additional_params_names = {
    ('TT_UNIMODAL', '--tt-sampling-strategy struct_ucb'): 'uni-tt',
    ('TT_UNIMODAL', '--tt-sampling-strategy ucb'): 'uni-tt-ucb',
    ('TT_UNIMODAL', '--tt-sampling-strategy ts'): 'uni-tt-ts',
    ('TT_UNIMODAL', '--tt-sampling-strategy struct_ucb --use-naive-z 1'): 'uni-tt-naive-z',
    ('TT_UNIMODAL', '--tt-sampling-strategy struct_ucb --use-fixed-design 1'): 'uni-tt-fixed-design',

    ('TT_BAI_STOP_UNI_SAMPLING', '--tt-sampling-strategy struct_ucb --max-iter 50000'): 'bai-stop-uni-sampling-struct',
    ('TT_BAI_STOP_UNI_SAMPLING', '--tt-sampling-strategy ucb --max-iter 50000'): 'bai-stop-uni-sampling-ucb',
    ('TT_BAI_STOP_UNI_SAMPLING', '--tt-sampling-strategy ts --max-iter 50000'): 'bai-stop-uni-sampling-ts',

    ('TT_BAI_SAMPLING_UNI_STOPPING', '--tt-sampling-strategy ucb'): 'bai-sampling-uni-stopping-ucb',
    ('TT_BAI_SAMPLING_UNI_STOPPING', '--tt-sampling-strategy ts'): 'bai-sampling-uni-stopping-ts',

    ('TT_BAI', '--tt-sampling-strategy ucb'): 'tt-bai-ucb',
    ('TT_BAI', '--tt-sampling-strategy ts'): 'tt-bai-ts',
}

fast_algos = ['Uniform', 'TT_BAI', 'TT_UNIMODAL', 'Unimodal-TaS']
is_fast_required = True

cmd = ''
c = 0
for config in cfg_list:
    for algo in algo_names:
        if is_fast_required and algo not in fast_algos:
            continue

        if algo in additional_params:
            for params in additional_params[algo]:
                cmd += (f'taskset -c {cores} python3.8 run.py '
                        f'--algo {algo} '
                        f'--delta {delta} '
                        f'--env-cfg configs/{config} '
                        f'--n-runs {n_runs} '
                        f'--n-jobs {n_jobs} '
                        f'--verbosity {verbosity} '
                        f'--results-dir results/{cfg_names[config]}/{additional_params_names[(algo, params)]} '
                        f'{params}')
                cmd += ' && '
                c += 1
        else:
            cmd += (f'taskset -c {cores} python3.8 run.py '
                    f'--algo {algo} '
                    f'--delta {delta} '
                    f'--env-cfg configs/{config} '
                    f'--n-runs {n_runs} '
                    f'--n-jobs {n_jobs} '
                    f'--verbosity {verbosity} '
                    f'--results-dir results/{cfg_names[config]}/{algo}')
            cmd += ' && '
            c += 1

cmd = cmd[:-4]

print(f'Count {c}')
text_file = open('run_cmd.txt', 'wt')
text_file.write(cmd)
text_file.close()
