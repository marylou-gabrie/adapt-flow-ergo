import numpy as np
import torch
from adaflonaco.phifour_utils import PhiFour


def run_MALA(target, x_init, n_steps, dt):
    '''
    target: target with the potentiel we will run the langevin on
    -> needs to have grad_U function implemented
    x (tensor): init points for the chains to update (batch_dim, dim)
    dt -> is multiplied by N for the phiFour before being passed
    '''

    assert isinstance(target, PhiFour)  # needs gradU method

    xs = []
    accs = []

    for t in range(n_steps):
        x = x_init.clone()
        x.detach_()

        x = x_init - dt * target.grad_U(x_init) 
        if dt > 0:
            x += dt * np.sqrt(2 / (dt * target.beta)) * torch.randn_like(x_init)

        ratio = - target.U(x) 
        ratio -= ((x_init - x + dt * target.grad_U(x)) ** 2 / (4 * dt)).sum(1)
        ratio += target.U(x_init) 
        ratio += ((x - x_init + dt * target.grad_U(x_init)) ** 2 / (4 * dt)).sum(1)
        ratio = target.beta * ratio
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
        x[~acc] = x_init[~acc]

        accs.append(acc)
        xs.append(x.clone())
        x_init = x.clone().detach()

    return torch.stack(xs), torch.stack(accs)


def run_metropolis(model, target, x_init, n_steps):
    xs = []
    accs = []

    for dt in range(n_steps):
        x = model.sample(x_init.shape[0])
        ratio = - target.beta * target.U(x) + model.nll(x)
        ratio += target.beta * target.U(x_init) - model.nll(x_init)
        ratio = torch.exp(ratio)
        u = torch.rand_like(ratio)
        acc = u < torch.min(ratio, torch.ones_like(ratio))
        x[~acc] = x_init[~acc]
        xs.append(x.clone())
        accs.append(acc)
        x_init = x.clone()

    return torch.stack(xs), torch.stack(accs)


def run_stochmetromalangevin(model, target, x_lang, n_steps, dt, lag=1):
    '''
    target: model with the potential we will run the MCMC for
    model: model for proposal with model.sample
    x_lang (tensor): init points for the chains to update (batch_dim, dim)
    dt -> will be multiplied by N for the phiFour
    lag (int): number of Langevin steps before considering resampling
    '''

    xs = []
    accs = []
    for t in range(n_steps):
        # Decide stochastically whether to update with MALA or NF
        if np.random.rand(1) < 0.5:
            x = model.sample(x_lang.shape[0])
            ratio = - target.beta * target.U(x) + model.nll(x)
            ratio += target.beta * target.U(x_lang) - model.nll(x_lang)
            ratio = torch.exp(ratio)
            u = torch.rand_like(ratio)
            acc = u < torch.min(ratio, torch.ones_like(ratio))
            x[~acc] = x_lang[~acc]
            accs.append(acc) 
            x_lang.data = x.clone()
        else:
            x = x_lang.clone()
            x = x_lang - dt * target.grad_U(x_lang) 
            if dt > 0:
                x += dt * np.sqrt(2 / (dt * target.beta)) * torch.randn_like(x_lang)
        
            ratio = - target.U(x) 
            ratio -= ((x_lang - x + dt * target.grad_U(x)) ** 2 / (4 * dt)).sum(1)
            ratio += target.U(x_lang) 
            ratio += ((x - x_lang + dt * target.grad_U(x_lang)) ** 2 / (4 * dt)).sum(1)
            ratio = target.beta * ratio
            ratio = torch.exp(ratio)
            u = torch.rand_like(ratio)
            acc = u < torch.min(ratio, torch.ones_like(ratio))
            x[~acc] = x_lang[~acc]
            x_lang.data = x.clone()

        xs.append(x_lang.clone())

    return torch.stack(xs), accs

# from https://github.com/jwalton3141/jwalton3141.github.io/blob/master/assets/posts/ESS/rwmh.py
# ref Gelman, Andrew, J. B. Carlin, Hal S. Stern, David B. Dunson, Aki Vehtari, and Donald B. Rubin. 2013. Bayesian Data Analysis. Third Edition. London: Chapman & Hall / CRC Press.


def compute_ESS(x):
    """
    Patching to take my convention of axis orders,
    and convert from torch to numpy
    x : (n_iter, m_chaines, dim)
    """
    try:
        x = x.detach().numpy()
    except AttributeError:
        x = x

    x = x.swapaxes(0, 1)
    return my_ESS(x)


def my_ESS(x):
    """
    Compute the effective sample size of estimand of interest.
    Vectorised implementation.
    x : m_chaines, n_iter, dim
    """
    if x.shape < (2,):
        raise ValueError(
            'Calculation of effective sample size'
            'requires multiple chains of the same length.')
    try:
        m_chains, n_iter = x.shape
    except ValueError:
        return [my_ESS(y.T) for y in x.T]

    def variogram(t): return (
        (x[:, t:] - x[:, :(n_iter - t)])**2).sum() / (m_chains * (n_iter - t))

    post_var = my_gelman_rubin(x)
    assert post_var > 0

    t = 1
    rho = np.ones(n_iter)
    negative_autocorr = False

    # Iterate until the sum of consecutive estimates of autocorrelation is negative
    while not negative_autocorr and (t < n_iter):
        rho[t] = 1 - variogram(t) / (2 * post_var)

        if not t % 2:
            negative_autocorr = sum(rho[t - 1:t + 1]) < 0

        t += 1

    return int(m_chains * n_iter / (1 + 2 * rho[1:t].sum()))


def my_gelman_rubin(x):
    """
    Estimate the marginal posterior variance. Vectorised implementation.
    x : m_chaines, n_iter
    """
    m_chains, n_iter = x.shape

    # Calculate between-chain variance
    B_over_n = ((np.mean(x, axis=1) - np.mean(x))**2).sum() / (m_chains - 1)

    # Calculate within-chain variances
    W = ((x - x.mean(axis=1, keepdims=True)) **
         2).sum() / (m_chains * (n_iter - 1))

    # (over) estimate of variance
    s2 = W * (n_iter - 1) / n_iter + B_over_n

    return s2
