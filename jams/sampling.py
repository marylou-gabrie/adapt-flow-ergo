import numpy as np
from realnvp.phifour_utils import PhiFour
import torch
import matplotlib.pyplot as plt
from realnvp.sampling import run_MALA


def jump_update(tilt_target, x, mode):
    choose_from = list(range(tilt_target.n_modes))
    choose_from.pop(mode)
    mode_prime = np.random.choice(choose_from)

    x_prime = tilt_target.sample_from_mode(mode_prime, x.shape[0])
    ratio = - tilt_target.U(x_prime, mode_prime) 
    ratio -= tilt_target.mix_comps[mode_prime].log_prob(x_prime) 
    ratio += tilt_target.U(x, mode) + tilt_target.mix_comps[mode].log_prob(x)
    ratio = torch.exp(ratio)
    # print(ratio)
    u = torch.rand_like(ratio)
    acc = u < torch.min(ratio, torch.ones_like(ratio))
    # print(acc)
    x_prime[~acc] = x[~acc]
    if ~acc[0]:
        mode_prime = mode

    return x_prime, mode_prime, acc


def run_jams(x_init, mode_init, target, tilt_target, n_steps, 
             alpha=0.1, dt=1e-4, occ_min=100, beta=1e-4):
    """
    - will always propose equally in all modes 
    - mode centers are kept fix 
    - only covariances are updated
    alpha (float): in (0,1) probability of jump move
    beta (float): regularization for cov
    occ_min: minimum occupation to start adapting covariance

    """
    # will collect index of mode visited at each iteration
    occ_modes = [] 
    xs = []
    accs_MALA = []
    accs_jump = []

    for it in range(n_steps):
        u = np.random.rand()
        if u > alpha:
            x, acc = run_MALA(target, x_init, 1, dt)
            x = x[-1, ...] #take last mala step
            mode = mode_init
            accs_MALA.append(acc)
        else:
            x, mode, acc = jump_update(tilt_target, x_init, mode_init)   
            accs_jump.append(acc)

        x_init = x.clone().requires_grad_()
        mode_init = mode
        
        xs.append(x.clone().detach())
        occ_modes.append(mode)

        occ_mode = ((torch.tensor(occ_modes) == mode) * 1).sum()
    
        if occ_mode > occ_min:
            with torch.no_grad():
                # new_mu = (torch.stack(xs)[torch.tensor(occ_modes) == mode]).mean(0)
                # tilt_target.mus[mode] = new_mu
                xs_ = torch.cat(xs)
                xs_centered = xs_[torch.tensor(occ_modes) == mode, :]

                # plt.figure('xs_centered')
                # plt.clf()
                # plt.title('mode ' +str(mode))
                # plt.plot(xs_centered.T)
                # plt.show(block=False)
                
                xs_centered -= tilt_target.mus[mode].view(1, tilt_target.dim)
                emp_cov = torch.einsum('ti,tj->ij', xs_centered, xs_centered) 
                emp_cov *= 1 / (xs_centered.shape[0])
                # emp_cov *= 2.38 ** 2 / (tilt_target.dim)
                new_cov = emp_cov + beta * torch.eye(tilt_target.dim,
                                                    device=tilt_target.device)
                
                tilt_target.covs[mode] = new_cov
                tilt_target.update_mixture_comps()
        
        if it % int(n_steps / 10) == 0:
            acc_jump_rate = (torch.tensor(accs_jump) * 1.).mean().item()
            acc_mala_rate = (torch.tensor(accs_MALA) * 1.).mean().item()
            print('it {:d} - accept jump: {:0.2e} - accept mala: {:0.2e}'.format(it, acc_jump_rate, acc_mala_rate), end='')

            if isinstance(target, PhiFour):
                xs_ = torch.cat(xs)
                rpos = ((xs_[:, int(tilt_target.dim / 2)] > 0) * 1.).mean().item()
                print('- rpos: {:0.2e}'.format(rpos), end='')
            
            print('')

    return xs, occ_modes, accs_MALA, accs_jump, tilt_target




# def run_MALA(target, x_init, n_steps, dt, mode=False): - no acceptance I am not sure if something is maybe wrong with the 
#     '''
#     target: target with the potentiel we will run the langevin on
#     -> needs to have grad_U function implemented
#     x (tensor): init points for the chains to update (batch_dim, dim)
#     dt -> is multiplied by N for the phiFour before being passed
#     '''

#     assert callable(getattr(target, 'grad_U', None))

#     xs = []
#     accs = []

#     for t in range(n_steps):
#         # x = x_init.clone()
#         # x.detach_().requires_grad_()

#         x_init_m = (x_init, mode) if mode is not None else x_init

#         x = x_init - dt * target.grad_U(*x_init_m) 
#         if dt > 0:
#             x += dt * np.sqrt(2 / dt) * torch.randn_like(x_init)

#         x_m = (x, mode) if mode is not None else x

#         ratio = - target.U(*x_m) 
#         ratio -= ((x_init - x + dt * target.grad_U(*x_m)) ** 2 / (4 * dt)).sum(1)
#         ratio += target.U(*x_init_m) 
#         ratio += ((x - x_init + dt * target.grad_U(*x_init_m)) ** 2 / (4 * dt)).sum(1)
#         ratio = torch.exp(ratio)
#         u = torch.rand_like(ratio)
#         acc = u < torch.min(ratio, torch.ones_like(ratio))
#         x[~acc] = x_init[~acc]
        
#         accs.append(acc)
#         xs.append(x.clone())
#         x_init = x.clone().detach().requires_grad_()
#     return torch.stack(xs), torch.stack(accs)



