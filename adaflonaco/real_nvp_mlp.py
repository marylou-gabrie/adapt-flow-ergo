import numpy as np
import torch
import torch.nn as nn
from adaflonaco.models import MLP, make_checker_mask
from torch.distributions.multivariate_normal import MultivariateNormal


class ResidualAffineCoupling(nn.Module):
    """ Affine Coupling layer arXiv:1605.08803

    Args:
        s (nn.Module): scale network
        t (nn.Module): translation network
        mask (binary tensor): mask for input
        dt (float): time step for the "integration"
    """

    def __init__(self, s=None, t=None, mask=None, dt=1):
        super(ResidualAffineCoupling, self).__init__()

        # checkboard
        self.mask = mask  # need input dimension info
        self.scale_net = s
        self.trans_net = t
        self.dt = dt

    def forward(self, x, log_det_jac=None, inverse=False):

        if log_det_jac is None:
            log_det_jac = 0

        s = self.mask * self.scale_net(x * (1 - self.mask))
        s = torch.tanh(s)
        t = self.mask * self.trans_net(x * (1 - self.mask))

        s = self.dt * s
        t = self.dt * t

        if inverse:
            if torch.isinf(s).any():
                raise RuntimeError('Scale factor has inf entries')
            if torch.isnan(torch.exp(-s)).any():
                raise RuntimeError('Scale factor has NaN entries')
            log_det_jac -= s.view(s.size(0), -1).sum(-1)

            x = x * torch.exp(-s) - t

        else:
            log_det_jac += s.view(s.size(0), -1).sum(-1)
            x = (x + t) * torch.exp(s)
            if torch.isnan(torch.exp(s)).any():
                raise RuntimeError('Scale factor has NaN entries')

        return x, log_det_jac


class RealNVP_MLP(nn.Module):
    """ Minimal Real NVP architecture

    Args:
        dims (int,):
        n_realnvp_blocks (int): each with 2 layers
        residual
        block_depth (int): number of pair of integration step per block
    """

    def __init__(self, dim, n_realnvp_blocks, block_depth,
                 init_weight_scale=1,
                 prior_arg={'type': 'standn'},
                 mask_type='half',  # 'half' or 'inter'
                 hidden_dim=10,
                 hidden_depth=3,
                 hidden_bias=True,
                 hidden_activation=torch.relu,
                 dim_phys=1,
                 device='cpu'):
        super(RealNVP_MLP, self).__init__()

        self.device = device
        self.dim = dim
        self.dim_phys = dim_phys
        self.n_blocks = n_realnvp_blocks
        self.block_depth = block_depth
        self.couplings_per_block = 2  # necessary for one update
        self.n_layers_in_coupling = hidden_depth
        self.hidden_dim_in_coupling = hidden_dim
        self.hidden_bias = hidden_bias
        self.hidden_activation = hidden_activation
        self.init_scale_in_coupling = init_weight_scale

        mask = torch.ones(dim, device=self.device)
        if mask_type == 'half':
            mask[:int(dim / 2)] = 0
        elif mask_type == 'inter':
            mask = make_checker_mask(dim, dim_phys, 0)
            mask = mask.to(device)
        else:
            raise RuntimeError('Mask type is either half or inter')
        
        self.mask = mask.view(1, dim)
        self.coupling_layers = self.initialize()

        self.prior_arg = prior_arg

        if prior_arg['type'] == 'standn':
            self.prior_prec = torch.eye(dim).to(device)
            self.prior_log_det = 0
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device), self.prior_prec)

        elif prior_arg['type'] == 'uncoupled':
            self.prior_prec = prior_arg['a'] * torch.eye(dim).to(device)
            self.prior_log_det = - torch.logdet(self.prior_prec)
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)

        elif prior_arg['type'] == 'coupled':
            self.beta_prior = prior_arg['beta']
            self.coef = prior_arg['alpha'] * dim
            prec = torch.eye(dim) * (3 * self.coef + 1 / self.coef)
            prec -= self.coef * torch.triu(torch.triu(torch.ones_like(prec),
                                                      diagonal=-1).T, diagonal=-1)
            prec = prior_arg['beta'] * prec
            self.prior_prec = prec.to(self.device)
            self.prior_log_det = - torch.logdet(prec)
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)
        
        elif prior_arg['type'] == 'white':
            cov = prior_arg['cov']
            self.prior_prec = torch.inverse(cov).to(device)
            self.prior_prec = 0.5 * (self.prior_prec + self.prior_prec.T)
            self.prior_mean = prior_arg['mean'].to(device)
            self.prior_log_det = - torch.logdet(self.prior_prec)
            self.prior_distrib = MultivariateNormal(
                prior_arg['mean'],
                precision_matrix=self.prior_prec
                )

        elif prior_arg['type'] == 'coupled_pbc':
            self.beta_prior = prior_arg['beta']
            dim_phys = prior_arg['dim_phys']
            dim_grid = prior_arg['dim_grid']
            
            eps = 0.1
            quadratic_coef = 4 + eps
            sub_prec = (1 + quadratic_coef) * torch.eye(dim_grid)
            sub_prec -= torch.triu(torch.triu(torch.ones_like(sub_prec),
                                                      diagonal=-1).T, diagonal=-1)
            sub_prec[0, -1] = - 1  # pbc
            sub_prec[-1, 0] = - 1  # pbc

            if dim_phys == 1:
                prec = prior_arg['beta'] * sub_prec

            elif dim_phys == 2:
                # interation along one axis
                prec = torch.block_diag(*(sub_prec for d in range(dim_grid)))
                # interation along second axis
                diags = torch.triu(torch.triu(torch.ones_like(prec),
                                                      diagonal=-dim_grid).T, diagonal=-dim_grid)
                diags -= torch.triu(torch.triu(torch.ones_like(prec),
                                                      diagonal=-dim_grid+1).T, diagonal=-dim_grid+1)
                prec -= diags
                prec[:dim_grid, -dim_grid:] = - torch.eye(dim_grid)  # pbc
                prec[-dim_grid:, :dim_grid] = - torch.eye(dim_grid)  # pbc
                prec = prior_arg['beta'] * prec

            self.prior_prec = prec.to(self.device)
            self.prior_log_det = - torch.logdet(prec)
            self.prior_distrib = MultivariateNormal(
                torch.zeros((dim,), device=self.device),
                precision_matrix=self.prior_prec)

        else:
            raise NotImplementedError("Invalid prior arg type")

    def forward(self, x, return_per_block=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        if return_per_block:
            xs = [x]
            log_det_jacs = [log_det_jac.clone()]

        for block in range(self.n_blocks):
            couplings = self.coupling_layers[block]

            for dt in range(self.block_depth):
                for coupling_layer in couplings:
                    x, log_det_jac = coupling_layer(x, log_det_jac)
                    if torch.isnan(x).any():
                        print('layer', dt)
                        raise RuntimeError('Layer became Nan')

                if return_per_block:
                    xs.append(x)
                    log_det_jacs.append(log_det_jac.clone())

        if return_per_block:
            return xs, log_det_jacs
        else:
            return x, log_det_jac

    def backward(self, x, return_per_block=False):
        log_det_jac = torch.zeros(x.shape[0], device=self.device)

        if return_per_block:
            xs = [x]
            log_det_jacs = [log_det_jac]

        for block in range(self.n_blocks):
            couplings = self.coupling_layers[::-1][block]

            for dt in range(self.block_depth):
                for coupling_layer in couplings[::-1]:
                    x, log_det_jac = coupling_layer(
                        x, log_det_jac, inverse=True)

                if return_per_block:
                    xs.append(x)
                    log_det_jacs.append(log_det_jac.clone())

        if return_per_block:
            return xs, log_det_jacs
        else:
            return x, log_det_jac

    def initialize(self):
        dim = self.dim
        coupling_layers = []

        for block in range(self.n_blocks):
            layer_dims = [self.hidden_dim_in_coupling] * \
                (self.n_layers_in_coupling - 2)
            layer_dims = [dim] + layer_dims + [dim]

            couplings = self.build_coupling_block(layer_dims)

            coupling_layers.append(nn.ModuleList(couplings))

        return nn.ModuleList(coupling_layers)

    def build_coupling_block(self, layer_dims=None):
        count = 0
        coupling_layers = []
        for count in range(self.couplings_per_block):
            
            s = MLP(layer_dims, init_scale=self.init_scale_in_coupling,
                    activation=self.hidden_activation, 
                    bias_bool=self.hidden_bias)
        
            t = MLP(layer_dims, init_scale=self.init_scale_in_coupling,
                    activation=self.hidden_activation, 
                    bias_bool=self.hidden_bias)

            s = s.to(self.device)

            t = t.to(self.device)

            if count % 2 == 0:
                mask = 1 - self.mask
            else:
                mask = self.mask

            dt = self.n_blocks * self.couplings_per_block * self.block_depth
            dt = 2 / dt

            coupling_layers.append(ResidualAffineCoupling(
                    s, t, mask, dt=dt))

        return coupling_layers

    def nll(self, x, from_z=False):
        """
        adding from_z option for 'reparametrization trick'
        """
        if from_z:
            z = x
            x, log_det_jac = self.forward(z)
            log_det_jac = - log_det_jac
        else:
            z, log_det_jac = self.backward(x)
        if self.prior_arg['type'] == 'white':
            z = z - self.prior_mean

        prior_ll = - 0.5 * torch.einsum('ki,ij,kj->k', z, self.prior_prec, z)
        prior_ll -= 0.5 * (self.dim * np.log(2 * np.pi) + self.prior_log_det)

        return - (prior_ll + log_det_jac)

    def sample(self, n):
        if self.prior_arg['type'] == 'standn':
            z = torch.randn(n, self.dim, device=self.device)
        else:
            z = self.prior_distrib.rsample(torch.Size([n, ])).to(self.device)

        return self.forward(z)[0]

    def U(self, x):
        """
        alias
        """ 
        return self.nll(x)
