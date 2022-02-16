import matplotlib
%matplotlib inline 
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]
    })

import os
import copy

import numpy as np
import time
import torch
import torch.nn.functional as F

from realnvp.phifour_utils import PhiFour
# from gaussian_utils import MoG, plot_2d_level, plot_2d_level_reversed
from realnvp.real_nvp_mlp import RealNVP_MLP
from realnvp.sampling import compute_ESS, run_haario
# from realnvp.sampling import compute_KS, run_mixedmetrolangevin
# from ergoflonaco.training import train
from realnvp.utils_io import get_file_name
from realnvp.utils_plots import (
    plot_map_point_cloud, 
    plot_map_point_cloud_Fourier,
    plot_Fourier_spectrum
)

from jams.sampling import run_jams

date = time.strftime('%d-%m-%Y')
random_id = str(np.random.randint(100))
print('random id!', random_id)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

dtype = torch.float32

ceph_home = '/mnt/ceph/users/mgabrie/nnmc/' 

## create JAMS chains
from jams.distributions import TiltedTarget
from jams.sampling import run_jams

from realnvp.phifour_utils import PhiFour
from realnvp.gaussian_utils import MoG
import realnvp.sampling

torch.manual_seed(25)

a = 0.1 
b = 0
N = dim = 100
beta = 20
dim_phys = 1

target = PhiFour(a, b, N, dim_phys=dim_phys, beta=beta, device=device) 

t = 5e-3

# initialize modes using langevin
mus = torch.load('/mnt/home/mgabrie/jams/temp/mus_N{:d}_beta{:0.2f}_a{:0.2f}.pyT'.format(dim, beta, a)).to(device)

# n_steps_init = 500
n_steps_init = 1000
x_init = mus.repeat_interleave(10, 0).clone().requires_grad_()
xs, accs = realnvp.sampling.run_MALA(target, x_init, n_steps_init, dt)
print('Accs were:', (accs * 1.).mean().item())

xs_reshape = torch.zeros(n_steps_init * 10, 2, dim, device=device)
xs_reshape[:, 0, :] = xs[:, :10, :].reshape(-1, dim)
xs_reshape[:, 1, :] = xs[:, 10:, :].reshape(-1, dim)
xs = xs_reshape
xs_centered = xs - mus.view(1, 2, dim)
covs = torch.einsum('tki,tkj->kij', xs_centered, xs_centered) / (xs_centered.shape[0])
covs += 1e-6 * torch.eye(dim, device=device).view(1, dim, dim)

tilt_target = TiltedTarget(
                dim, target_logpdf= lambda x: - beta * target.U(x),
                mus_init=mus, 
                ws_init=0.5 * torch.ones(2, device=device),
                covs_init=covs,
                device=device
                )
tilt_target.update_mixture_comps()

x = tilt_target.sample_from_mode(0, 10)
plt.figure()
plt.title('Sampling from one mode at initialization')
plt.plot(x.view(-1, dim).T.detach().cpu())
plt.plot(mus.T.detach().cpu(), c='k')
plt.show(block=False)


times = []

mode_init = 0
n_steps = int(2e5)
x_init = tilt_target.sample_from_mode(mode=mode_init, n_sample=1)
x_init.requires_grad_()

for rep in range(1):
    start = time.time()
    xs, occ_modes, accs_mala, accs_jump, tilt_target = run_jams(x_init, mode_init,
                                        target, 
                                        tilt_target, n_steps, 
                                        alpha=0.1, dt=dt, 
                                        occ_min=3000, beta=1e-4)
    ellapsed = time.time() - start
    times.append(ellapsed)
    print('ellapsed', ellapsed)
    
    x_init = xs[-1].clone().requires_grad_()
    xs = torch.cat(xs).cpu().numpy()
    

    plt.figure()
    plt.plot(xs[-100:].T)
    plt.show(block=False)
    
    ceph_home = '/mnt/ceph/users/mgabrie/ergo/' 
    save_at = ceph_home + 'mala/Jams_accs_mala_n{:0.1e}_{:d}_rd{:d}.pyT'.format(n_steps, rep, random_id)
    torch.save(accs_mala, save_at)
    print(save_at)
    save_at = ceph_home + 'mala/Jams_accs_jumps_n{:0.1e}_{:d}_rd{:d}.pyT'.format(n_steps, rep, random_id)
    torch.save(accs_jump, save_at)
    print(save_at)
    save_at = ceph_home + 'mala/Jams_xs_n{:0.1e}_{:d}_rd{:d}.pyT'.format(n_steps, rep, random_id)
    torch.save(xs, save_at)
    print(save_at)

save_at =  ceph_home + 'mala/Jams_times_n{:0.1e}_rd{:d}.pyT'.format(n_steps)
torch.save(torch.tensor(ellapsed), save_at)
print(save_at)