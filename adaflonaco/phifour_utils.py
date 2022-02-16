import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.multivariate_normal import MultivariateNormal


class PhiFour(nn.Module):
    def __init__(self, a, b, dim_grid, dim_phys=1,
                 beta=1,
                 bc=('dirichlet', 0),
                 tilt=None,
                 device='cpu'):
        """
        Class to handle operations around PhiFour model
        Args:
            a: coupling term coef
            b: local field coef
            dim_grid: grid size in one dimension
            dim_phys: number of dimensions of the physical grid
            beta: inverse temperature
            tilt: None or {"val":0.7, "lambda":0.1} - for biasing distribution
        """
        self.device = device

        self.a = a
        self.b = b
        self.beta = beta
        self.dim_grid = dim_grid
        self.dim_phys = dim_phys
        self.sum_dims = tuple(i + 1 for i in range(dim_phys))

        self.bc = bc
        self.tilt = tilt

    def init_field(self, n_or_values):
        if isinstance(n_or_values, int):
            x = torch.rand((n_or_values,) + (self.dim_grid,) * self.dim_phys)
            x = x * 2 - 1
        else:
            x = n_or_values
        return x
    
    def reshape_to_dimphys(self, x):
        if self.dim_phys == 2:
            x_ = x.reshape(-1, self.dim_grid, self.dim_grid)
        else:
            x_ = x
        return x_

    def V(self, x):
        x = self.reshape_to_dimphys(x)
        coef = self.a * self.dim_grid
        V = ((1 - x ** 2) ** 2 / 4 + self.b * x).sum(self.sum_dims) / coef
        if self.tilt is not None: 
            tilt = (self.tilt['val'] - x.mean(self.sum_dims)) ** 2 
            tilt = self.tilt["lambda"] * tilt / (4 * self.dim_grid)
            V += tilt
        return V

    def U(self, x):
        # Does not include the temperature! need to be explicitely added in Gibbs factor
        assert self.dim_phys < 3
        x = self.reshape_to_dimphys(x)

        if self.bc[0] == 'dirichlet':
            x_ = F.pad(input=x, pad=(1,) * (2 * self.dim_phys), mode='constant',
                      value=self.bc[1])
        elif self.bc[0] == 'pbc':
            #adding "channel dimension" for circular torch padding 
            x_ = x.unsqueeze(0) 
            #only pad one side, not to double count gradients at the edges
            x_ = F.pad(input=x_, pad=(1,0,) * (self.dim_phys), mode='circular')
            x_.squeeze_(0) 
        else:
            raise NotImplementedError("Only dirichlet and periodic BC"         
                                      "implemeted for now")

        if self.dim_phys == 2:
            grad_x = ((x_[:, 1:, :-1] - x_[:, :-1, :-1]) ** 2 / 2)
            grad_y = ((x_[:, :-1, 1:] - x_[:, :-1, :-1]) ** 2 / 2)
            grad_term = (grad_x + grad_y).sum(self.sum_dims)
        else:
            grad_term = ((x_[:, 1:] - x_[:, :-1]) ** 2 / 2).sum(self.sum_dims)
        
        coef = self.a * self.dim_grid
        return grad_term * coef + self.V(x) 

    def grad_U(self, x_init):
        x = x_init.detach()
        x = x.requires_grad_()
        optimizer = torch.optim.SGD([x], lr=0)
        optimizer.zero_grad()
        loss = self.U(x).sum()
        loss.backward()
        return x.grad.data