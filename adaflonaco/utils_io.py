ceph_home = 'temp/'

def get_file_name(args_target,  
                  args_training, 
                  args_model=None,
                  date='', 
                  random_id='',
                  plus='',
                  ceph_home=ceph_home):

    folder = ceph_home + 'models/' + args_target['type']

    name = date + '_' + args_training['args_samp']['samp'] + '_' 
    name += args_target['type'] 

    if args_target['type'] == 'phi4':
        N, a, b, beta, tilt = (args_target[key] for key in ['N', 'a', 'b', 'beta', 'tilt'])
        langevin_ratio_pos_init = args_training['args_samp']['ratio_pos_init']
        prior = args_model['args_prior']['type']

        name += '_N{:d}_a{:0.2f}_b{:0.2e}_beta{:0.2f}'.format(N, a, b, beta)
        if tilt is not None:
            name += '_tv{:0.2f}'.format(tilt['val'])
        name += '_prior{:s}'.format(prior)
        name += '_rposinit{:s}'.format(str(langevin_ratio_pos_init))

    if args_model is not None:
        n_realnvp_block, block_depth = (args_model[key] for key in ['n_realnvp_block', 'block_depth'])
        name += '_{:d}blocks{:d}deep'.format(n_realnvp_block, block_depth)

    name += '_{:s}_{:s}.pyT'.format(plus, random_id)

    return folder + '/' + name