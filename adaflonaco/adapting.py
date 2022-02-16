import copy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
import torch
from torch.nn.utils import clip_grad_norm_
from adaflonaco.phifour_utils import PhiFour
from adaflonaco.sampling import (
    run_MALA,
    run_metropolis,
    run_stochmetromalangevin,
    compute_ESS
)
from memory_profiler import profile


def mcmc_adapt(model, target, n_iter=10, lr=1e-1, bs=100,
               use_scheduler=False,
               step_schedule=10000,
               args_samp={'samp': 'direct'},
               estimate_tau=False,
               return_all_xs=True,
               jump_tol=1e2,
               save_splits=10,
               grad_clip=1e4,
               ):
    """"
     Args:
        model (Realnvp_MLP)
        target (MoG, PhiFour, RandomPeriodic)
        n_iter (int)
        lr (float): learning rate
        bs (int): batchsize
        args_samp: dic 
            -> 'samp': 'mh', 'metromalalangevin'
            -> 'n_steps_burnin': 
            -> 'dt':
            -> 'n_tot':
    """

    # setting the loss
    def loss_func(x): return (model.nll(x) - target.U(x)).mean()
            
    # Getting that out of the way
    if isinstance(target, PhiFour):
        dimx = target.dim_grid * target.dim_phys

    # setting the sampling
    if 'langevin' in args_samp['samp']:
        skip_burnin = False
        assert args_samp['n_tot'] <= bs
        
        if args_samp['x_init_samp'] is not None:
            x_init = args_samp['x_init_samp'][-args_samp['n_tot']:]
            skip_burnin = True
        
        elif isinstance(target, PhiFour):
            x_init = torch.ones(args_samp['n_tot'], model.dim, device=model.device)
            n_pos = int(args_samp['ratio_pos_init'] * args_samp['n_tot'])
            if target.tilt is None:
                x_init[n_pos:, :] = -1
            else:
                n_tilt = int(target.tilt['val'] * model.dim)
                x_init[n_pos:, n_tilt:] = -1
                x_init[:n_pos, :(model.dim - n_tilt)] = -1

        else:
            raise NotImplementedError("That target class is not implemented")

        x_init = x_init.detach().requires_grad_()
        kwargs = {}

        if not skip_burnin: 
            bs_burnin = int(args_samp['n_steps_burnin'] * x_init.shape[0])
            start = time.time()
            n_steps = int(bs_burnin / x_init.shape[0])
            x, accs = run_MALA(target, x_init, n_steps,
                               dt=args_samp['dt'] * model.dim)
            print('MALA burnin done! time: {:f}s - accs {:0.2f}'.format(
                time.time() - start,
                (accs.cpu().numpy() * 1).mean())
                )
            x_init = x[-1, ...].detach().requires_grad_()

        if args_samp['samp'] == 'stochmetromalangevin':

            def sample_func(bs, x_init=x_init, dt=100, acc_rate=None):
                n_steps = int(bs / x_init[0].shape[0])
                x, acc = run_stochmetromalangevin(
                    model, target, x_init, n_steps, dt=dt * dimx)
                kwargs['x_init'] = x[-1, ...].detach().requires_grad_()
                return x
        else:
            raise NotImplementedError("Sampling config not understood")

        kwargs = {'x_init': x_init,
                  'dt': args_samp['dt']}


    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                    step_size=step_schedule, 
                                                    gamma=0.5)
    else:
        use_scheduler = False

    # preparing plots
    fig = plt.figure(figsize=(12, 5))
    gs = gridspec.GridSpec(2, 5)
    axs = [fig.add_subplot(gs[a]) for a in range(10)] 
    a = 0

    # preparing logs
    xs = []
    losses = []
    models = [copy.deepcopy(model)]
    taus = []
    acc_rates = []
    acc_rates_mala = []
    grad_norms = []
    times = []

    for t in range(n_iter):
        start = time.time()
        optimizer.zero_grad()

        x_ = sample_func(bs, **kwargs)
        
        x = x_.reshape(-1, dimx).detach()
        loss = loss_func(x)

        if t > 0 and loss - losses[-1] > jump_tol:
            print('KL wants to jump, terminating learning')
            break

        loss.backward()
        clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        # logging
        if return_all_xs or t % (n_iter / 10) == 0:
            xs.append(x_)

        times.append(time.time() - start)
        losses.append(loss.item())

        total_norm = 0
        for l,p in enumerate(model.parameters()):
            try:
                param_norm = p.grad.detach().data.norm(2)
                total_norm += param_norm.item() ** 2
            except AttributeError:
                if t == 0:
                    print('No grad for parameters or last or fist translation with uniform priir')
            if torch.isnan(param_norm).any():
                raise RuntimeError('Gradients block {:d} has NaN'.format(l))
        total_norm = total_norm ** 0.5     
        grad_norms.append(total_norm)

        if use_scheduler:
            scheduler.step()

        if estimate_tau:
            tau = x.shape[0] * x.shape[1] / \
                np.mean(compute_ESS(x.detach().cpu()))
            taus.append(tau)

        x_last = x.clone()
        _, acc = run_metropolis(model, target, x_last, 1)
        acc_rate = (acc.cpu().numpy() * 1).mean()
        acc_rates.append(acc_rate.item())

        x_last = x.clone()
        _, acc = run_MALA(target, x_last, 1,  dt=args_samp['dt'] * model.dim)
        acc_rate = (acc.cpu().numpy() * 1).mean()
        acc_rates_mala.append(acc_rate.item())

        if t % (n_iter / save_splits) == 0 or n_iter <= save_splits:
            models.append(copy.deepcopy(model))

            print('t={:0.1e}'.format(t),
                  'Loss: {:3.2f}'.format(loss.item()), end=' \t')

            print('mh:', acc_rates[-1], 'mala:', acc_rates_mala[-1], end='\t')

            print('Gd: {:0.0e}'.format(total_norm), end='\t')
            
            for param_group in optimizer.param_groups:
                print('lr: {:0.2e}'.format(param_group['lr']), end='\t')

            # compute fraction with mean > 0 or mean < 1
            if isinstance(target, PhiFour):
                x_gen = model.sample(bs)
                frac_pos = (x_gen.mean(1) > 0).sum() / \
                    float(x_gen.shape[0])
                print('Frac gen pos: {:0.2f}'.format(frac_pos.item()), end='\t')
            
            print('')

        if t % (n_iter / 10) == 0:
            if isinstance(target, PhiFour) and target.dim_phys == 1:
                plt.sca(axs[a])
                plt.title('t= ' + str(t))
                x_gen = model.sample(xs[-1].shape[1])
                for i in range(xs[-1].shape[1]):
                    plt.plot(xs[-1][-1,  i, :].detach().cpu(),
                             c='b', alpha=0.2)
                for i in range(x_gen.shape[0]):
                    plt.plot(x_gen[i, :].detach().cpu(),
                             c='k', alpha=0.2)

            plt.tight_layout()
            a += 1

    to_return = {
        'model': model,
        'losses': losses,
        'xs': xs,
        'models': models,
        'taus': taus,
        'acc_rates': acc_rates,
        'acc_rates_mala': acc_rates_mala,
        'grad_norms': grad_norms,
        'times': times
    }

    return to_return
