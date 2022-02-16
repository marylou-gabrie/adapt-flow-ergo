import copy
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.shape_base import block
import torch
import torch.nn.functional as F

from adaflonaco.phifour_utils import PhiFour
from adaflonaco.real_nvp_mlp import RealNVP_MLP
from adaflonaco.adapting import mcmc_adapt
from adaflonaco.utils_plots import (
    moving_average
)


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

N = 100 
beta = 20
dim_phys = 1
a = 0.1
b = 0.0

phi4 = PhiFour(a, b, N, dim_phys=dim_phys, beta=beta) 

dim = N * dim_phys
n_realnvp_block = 4
block_depth = 1

model = RealNVP_MLP(dim, n_realnvp_block, block_depth, 
                    init_weight_scale=1e-6,
                    prior_arg={'type': 'coupled', 'alpha': a, 'beta': beta},
                    mask_type='inter',
                    )

args_samp = {
    'samp': 'stochmetromalangevin', 'dt': 5e-5, 
    'n_steps_burnin': 1e3,
    'n_tot': 100, 
    'ratio_pos_init': 0.5,
    'x_init_samp': None
    }

bs = int(1e2)

_ = mcmc_adapt(model, phi4, n_iter=int(1e1),
    lr=1e-3, bs=bs,
    args_samp=args_samp,
    )

fig = plt.figure(figsize=(12, 4))
axs = [plt.subplot(1, 3, i+1) for i in range(3)]

plt.sca(axs[0])
plt.plot(_['losses'])
plt.xlabel('iter')
plt.ylabel('KL')

wdw = 50
plt.sca(axs[1])
plt.plot(_['acc_rates'], alpha=0.1)
accs_mva = moving_average(_['acc_rates'], wdw)
plt.plot(np.arange(len(_['acc_rates']) - wdw + 1) + wdw, accs_mva, c='C0')
plt.xlabel('Training iterations')
plt.ylabel('Acceptance ratio')

wdw = 20
plt.sca(axs[2])
plt.plot(_['grad_norms'], alpha=0.1)
taus_mva = moving_average(_['grad_norms'], wdw)
plt.plot(np.arange(len(_['grad_norms']) - wdw + 1) + wdw, taus_mva, c='C0')
plt.xlabel('Training iterations')
plt.ylabel('Gradnorms')
plt.show(block=False)
        
plt.show(block=False)