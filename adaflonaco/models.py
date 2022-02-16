import torch
import torch.nn as nn
import numpy as np


def make_checker_mask(dim, dim_phys, parity):
    if dim_phys == 1:
        checker = torch.ones((dim,), dtype=torch.uint8) - parity
        checker[::2] = parity
    elif dim_phys == 2:
        dim_grid = int(np.sqrt(dim))
        checker = torch.ones((dim_grid, dim_grid), dtype=torch.uint8) - parity
        checker[::2] = parity
        checker[::2, ::2] = parity
        checker[1::2, 1::2] = parity
    else:
        raise RuntimeError('Mask shape not understood')
    return checker.float()


class MLP(nn.Module):
    def __init__(self, layerdims, activation=torch.relu, init_scale=None,
                 bias_bool=True):
        super(MLP, self).__init__()
        self.layerdims = layerdims
        self.activation = activation
        linears = [nn.Linear(layerdims[i], layerdims[i + 1], bias=bias_bool) for i in range(len(layerdims) - 1)]
        
        if init_scale is not None:
            for l, layer in enumerate(linears):
                torch.nn.init.normal_(layer.weight, 
                                      std=init_scale/np.sqrt(layerdims[l]))
                if bias_bool:
                    torch.nn.init.zeros_(layer.bias)

        self.linears = nn.ModuleList(linears)

    def forward(self, x):
        layers = list(enumerate(self.linears))
        for _, l in layers[:-1]:
            x = self.activation(l(x))
        y = layers[-1][1](x)
        return y