import multiprocessing

params = {
    # experiment modes
    'env_version': 1,
    'gradient': False,
    'sampling': True,
    #
    'seeds': list(range(20)),
    #
    'flamda': 1,
    'lamdas': [],
    'fgamma': .9,
    'gammas': [],
    'freg': 'L2',
    'regs': [],
    #
    'feta': 1,
    'etas': [],
    # 
    'fn_sample': 200,
    'n_samples': [20, 50, 100, 200, 500, 1000],
    #
    'max_iterations': 1000,
    #
    'n_jobs': multiprocessing.cpu_count()
}

if params['env_version'] == 1:
    params['eps'] = .3
    params['fbeta'] = 5
    params['betas'] = []
else:
    params['eps'] = .2
    params['fbeta'] = 1
    params['betas'] = [1, 2, 3, 5, 10]