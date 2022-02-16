import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal


class TiltedTarget(nn.Module):
    """
    Implemented for Gaussian auxiliary distributions - 
    Eq (2.1) in paper
    """
    def __init__(self, dim, target_logpdf, ws_init=None, mus_init=None, 
                 covs_init=None, device='cpu'):

        self.device = device
        self.dim = dim
        self.target_logpdf = target_logpdf

        self.ws = ws_init
        self.mus = mus_init
        self.covs = covs_init
        if ws_init is not None:
            self.n_modes = len(ws_init)
        # self.covs_inv = torch.linalg.inv(covs_init)
    
    def init_params(self, mu_init, cov_coef=1):
        self.mus = mu_init.to(self.device)
        self.ws = torch.array([1 / len(mu_init) for m in mu_init],
                      device=self.device)
        self.covs = torch.stack([torch.eye(self.dim) for m in mu_init])
        self.covs = self.covs.to(self.device) * cov_coef
        self.n_modes = len(mu_init)
        # self.covs_inv = torch.linalg.inv(self.covs)

    def update_mixture_comps(self):
        self.mix_comps = []
        for m, mu in enumerate(self.mus):
            comp =  MultivariateNormal(
                mu,
                covariance_matrix=self.covs[m])
            self.mix_comps.append(comp)

    def sample_from_mode(self, mode, n_sample):
        return self.mix_comps[mode].rsample(torch.Size([n_sample,]))

    def logpdf(self, x, mode):
        assert mode < len(self.mus)
        logpdf = self.target_logpdf(x) + torch.log(self.ws[mode]) 
        logpdf += self.mix_comps[mode].log_prob(x) 
        
        denom = []
        for m in range(len(self.mus)):
            denom.append(self.mix_comps[m].log_prob(x) + torch.log(self.ws[m]))

        logpdf -= torch.logsumexp(torch.tensor(denom), 0) 
        return logpdf

    def U(self, x, mode):
        return - self.logpdf(x, mode)

    def grad_U(self, x, mode):
        return torch.autograd.grad(self.U(x, mode), x)[0]