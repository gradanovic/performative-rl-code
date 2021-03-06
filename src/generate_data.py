from statistics import mean, stdev
import time
import itertools
import json
from joblib import Parallel, delayed
from tqdm import tqdm

from src.performative_predictionV1 import Performative_PredictionV1
from src.performative_predictionV2 import Performative_PredictionV2
from src.envs.gridworldV1 import GridworldV1
from src.envs.gridworldV2 import GridworldV2
from src.utils import *


def generate_data(params):
    """"
    """
    print('Beging generating performative prediction data\n')
    start = time.time()

    # Load Experiment Mode
    env_version = params['env_version']
    gradient = params['gradient']
    sampling = params['sampling']
    # Load Experiment Parameters
    n_jobs = params['n_jobs']
    # environment parameters
    eps = params['eps']
    fbeta = params['fbeta']
    betas = params['betas']
    fgamma = params['fgamma']
    gammas = params['gammas']
    # perormative prediction parameters
    max_iterations = params['max_iterations']
    flamda = params['flamda']
    lamdas = params['lamdas']
    freg = params['freg']
    regs = params['regs']
    # gradient parameters
    feta = params['feta']
    etas = params['etas']
    # sampling parameters
    seeds = params['seeds']
    fn_sample = params['fn_sample']
    n_samples = params['n_samples']

    # Prepare Experiment Configurations
    configs = []
    if not gradient and not sampling:
        # iterate lamdas
        for lamda in lamdas:
            configs.append({'beta': fbeta, 'lamda': lamda, 'gamma': fgamma, 'reg': freg})
        # iterate betas
        for beta in betas:
            configs.append({'beta': beta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg})
        # iterate regs
        for reg in regs:
            configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': reg})
        # iterate gammas
        for gamma in gammas:
            configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': gamma, 'reg': freg})
        # iterate gammas and lamdas
        for lamda, gamma in itertools.product(lamdas, gammas):
            configs.append({'beta': fbeta, 'lamda': lamda, 'gamma': gamma, 'reg': freg})
    if gradient:
        # iterate etas
        assert freg == 'L2'
        for eta in etas:
            if sampling:
                configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'eta': eta, 'n_sample': fn_sample})
            else:
                configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'eta': eta})
    if sampling:
        # iterate n_samples
        for n_sample in n_samples:
            if gradient:
                configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'eta': feta, 'n_sample': n_sample})
            else:
                configs.append({'beta': fbeta, 'lamda': flamda, 'gamma': fgamma, 'reg': freg, 'n_sample': n_sample})
    # remove duplicates
    configs = [dict(tup) for tup in set(tuple(d.items()) for d in configs)]
    
    # Generate Output
    if not sampling:
        # parallelize over configurations
        with tqdm_joblib(tqdm(desc="Executing Performative Prediction", total=len(configs))) as progress_bar:
            outputs = Parallel(n_jobs=min(n_jobs, len(configs)))(
                delayed(execute_performative_prediction)(config, env_version, eps, max_iterations, gradient, sampling)
                for config in configs
            )
    else:
        outputs = []
        # parallelize over seeds
        configs = sorted(configs, key=lambda d: d['n_sample']) 
        for config in configs:
            output = {k: v for k, v in config.items()}
            with tqdm_joblib(tqdm(desc=f"Executing Performative Prediction for n_sample={config['n_sample']}", total=len(seeds))) as progress_bar:
                tmp_output = Parallel(n_jobs=min(n_jobs, len(seeds)))(
                    delayed(execute_performative_prediction)(config, env_version, eps, max_iterations, gradient, sampling, seed)
                    for seed in seeds
                )
            d_diffs = [tmp_output[seed]['d_diff'] for seed in seeds]
            output['d_diff_mean'] = list(map(mean, zip(*d_diffs)))
            output['d_diff_std'] = list(map(stdev, zip(*d_diffs)))
            if gradient:
                sub_gaps = [tmp_output[seed]['sub_gap'] for seed in seeds]
                output['sub_gap_mean'] = list(map(mean, zip(*sub_gaps)))
                output['sub_gap_std'] = list(map(stdev, zip(*sub_gaps)))
            outputs.append(output)

    # Store Output
    with open(f'data/outputs.json', 'w') as f:
        json.dump(outputs, f, indent=4)

    end = time.time()
    print(f'Time elapsed: {end - start}')
    print('Finish generating data\n')

    return

def execute_performative_prediction(config, env_version, eps, max_iterations, gradient, sampling, seed=1):
    """
    """
    beta = config['beta']
    lamda = config['lamda']
    gamma = config['gamma']
    reg = config['reg']
    if gradient: eta = config['eta']
    else: eta = None
    if sampling: n_sample = config['n_sample']
    else: n_sample = None

    if env_version == 1:
        env = GridworldV1(beta, eps, gamma, sampling, n_sample, seed)
        algorithm = Performative_PredictionV1(env, max_iterations, lamda, reg, gradient, eta, sampling)
    elif env_version == 2:
        env = GridworldV2(beta, eps, gamma, sampling, n_sample, seed)
        algorithm = Performative_PredictionV2(env, max_iterations, lamda, reg, gradient, eta, sampling)
    else:
        raise ValueError()

    output = {k: v for k,v in config.items()}
    algorithm.execute()
    output['d_diff'] = algorithm.d_diff
    if gradient:
        output['sub_gap'] = algorithm.sub_gap

    
    # store initial visualization of env
    vis = env._get_env_vis()
    config_name = "limit_" + f"beta={beta}_lambda={lamda}_gamma={gamma}_reg={reg}"
    if gradient: config_name += f"eta={eta}"
    if sampling:
        config_name += f"n_sample={n_sample}_seed={seed}"
    with open(f'limiting_envs/{config_name}.json', 'w') as f:
        json.dump(vis, f, indent=4)

    return output