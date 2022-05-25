import click
import multiprocessing

from src.generate_data import generate_data
from src.generate_plots import generate_plots


@click.command()
# experiment modes
@click.option('--env_version', default=1, type=click.Choice([1, 2]), help='Environment version')
@click.option('--gradient', is_flag=True, help='Flag for repeated gradient ascent method')
@click.option('--sampling', is_flag=True, help='Flag for finite samples')
# experiment parameters
@click.option('--eps', default=.3, type=float, help='Environment parameter epsilon')
@click.option('--fbeta', default=5, type=float, help='Fixed value for (smoothness) parameter beta')
@click.option('--betas', multiple=True, default=[.1, 1, 5, 10, 100], type=float, help='List of values for (smoothness) parameter beta')
@click.option('--flamda', default=1, type=float, help='Fixed value for (regularization) parameter lambda')
@click.option('--lamdas', multiple=True, default=[0, .1, 0.5, 1, 5], type=float, help='List of values for (regularization) parameter lambda')
@click.option('--fgamma', default=.9, type=float, help='Fixed value for discount factor gamma')
@click.option('--gammas', multiple=True, default=[], type=float, help='List of values for discount factor gamma')
@click.option('--freg', default='L2', type=click.Choice(['L2', 'ER']), help='Fixed value for regularizer')
@click.option('--regs', multiple=True, default=[], type=click.Choice(['L2', 'ER']), help='List of values for regularizer')
# gradient
@click.option('--feta', default=1, type=float, help='Fixed value for (step size) parameter eta')
@click.option('--etas', multiple=True, default=[.05, .1, .2, 1, 2], type=float, help='List of values for (step size) parameter eta')
# sampling
@click.option('--fn_sample', default=200, type=int, help='Fixed value for number of samples')
@click.option('--n_samples', multiple=True, default=[20, 50, 100, 200, 500, 1000], type=int, help='List of values for number of samples')
@click.option('--seeds', multiple=True, default=list(range(20)), type=int, help='List of experiment seeds')
# iterations
@click.option('--max_iterations', default=1000, type=int, help='Number of Iterations')
# n_jobs
@click.option('--n_jobs', default=multiprocessing.cpu_count(), type=int, help='Number of jobs')
def run_experiment(env_version, gradient, sampling, eps, fbeta, betas, flamda, lamdas, fgamma, gammas, freg, regs, feta, etas, fn_sample, n_samples, seeds, max_iterations, n_jobs):

    print("Begin experiment\n")

    params = {}
    params['env_version'] = env_version
    params['gradient'] = gradient
    params['sampling'] = sampling
    params['eps'] = eps
    params['fbeta'] = fbeta
    params['betas'] = betas
    params['flamda'] = flamda
    params['lamdas'] = lamdas
    params['fgamma'] = fgamma
    params['gammas'] = gammas
    params['freg'] = freg
    params['regs'] = regs
    params['feta'] = feta
    params['etas'] = etas
    params['fn_sample'] = fn_sample
    params['n_samples'] = n_samples
    params['seeds'] = seeds
    params['max_iterations'] = max_iterations
    params['n_jobs'] = n_jobs

    generate_data(params)
    generate_plots(params)

    print("Finish experiment")

if __name__ == '__main__':
    
    run_experiment()