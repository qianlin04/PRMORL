from types import SimpleNamespace


HYPERPARAMS = {
       

    'HalfCheetah_MOOF_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MOOF-HalfCheetah-v2", #
        'num_objective':    2, #
        'cuda':             True,
        'load_model':             False,
        'use_angleterm':        True,
        'name':             'MOOF_TD3_HER', #
        'replay_size':      5000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.002,  #
        'w_bc_size':        0.05,    #
        'w_bc_min':         0.0,
        'weight_num':       4, #
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995,#
        'weight_bc_reward': 10,
        'batch_size':       256,
        'process_count':    1, #
        'eval_freq':        2e5,#
        'eval_interp_freq': 1000, #
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      10,   #
        'eval_episodes':    3,   #
        'num_eval_env':     10,
        'max_episode_len':  1000, #
        'layer_N_critic':   2,
        'layer_N_actor':    2,
        'angle_loss_coeff':    10,
        'hidden_size':      400
    }),

    'Hopper_MOOF_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MOOF-Hopper-v2",
        'num_objective':    2,
        'cuda':             True,
        'load_model':             False,
        'use_angleterm':        True,
        'name':             'MOOF_TD3_HER',
        'replay_size':      5000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.002,
        'w_bc_size':        0.05,
        'w_bc_min':         0.0,
        'weight_num':       4,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995, 
        'weight_bc_reward': 10,
        'batch_size':       256,
        'process_count':    1,
        'eval_freq':        2e5,
        'eval_interp_freq': 1000,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      20,
        'eval_episodes':    3, 
        'num_eval_env':     10,
        'max_episode_len':  1000,  #
        'layer_N_critic':   2,
        'layer_N_actor':    2,
        'angle_loss_coeff':    10,
        'hidden_size':      400
    }),

    'Walker2d_MOOF_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MOOF-Walker2d-v2",
        'num_objective':    2,
        'cuda':             True,
        'load_model':             False,
        'use_angleterm':        True,
        'name':             'MOOF_TD3_HER',
        'replay_size':      5000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.002,
        'w_bc_size':        0.05,
        'w_bc_min':         0.0,
        'weight_num':       4,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995,
        'weight_bc_reward': 10,
        'batch_size':       256,
        'process_count':    1,
        'eval_freq':        2e5,
        'eval_interp_freq': 1000,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      10,
        'eval_episodes':    3,
        'num_eval_env':     10,
        'max_episode_len':  1000,#1000 for d4rl, 500 for d4morl
        'layer_N_critic':   2,
        'layer_N_actor':    2,
        'angle_loss_coeff':    10,
        'hidden_size':      400
    }),

    'Swimmer_MOOF_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MOOF-Swimmer-v2",
        'num_objective':    2,
        'cuda':             True,
        'load_model':             False,
        'use_angleterm':        True,
        'name':             'MOOF_TD3_HER',
        'replay_size':      5000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.002,
        'w_bc_size':        0.05,
        'w_bc_min':         0.0,
        'weight_num':       4,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995, 
        'weight_bc_reward': 10,
        'batch_size':       256,
        'process_count':    1,
        'eval_freq':        2e5,
        'eval_interp_freq': 1000,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      20,
        'eval_episodes':    3, 
        'num_eval_env':     10,
        'max_episode_len':  500,  #
        'layer_N_critic':   2,
        'layer_N_actor':    2,
        'angle_loss_coeff':    10,
        'hidden_size':      400
    }),

    'Ant_MOOF_TD3_HER': SimpleNamespace(**{
        'scenario_name':         "MOOF-Ant-v2",
        'num_objective':    2,
        'cuda':             True,
        'load_model':             False,
        'use_angleterm':        True,
        'name':             'MOOF_TD3_HER',
        'replay_size':      5000000,
        'time_steps':       1000000,
        'start_timesteps':  10000,
        'w_step_size':      0.002,
        'w_bc_size':        0.05,
        'w_bc_min':         0.0,
        'weight_num':       4,
        'expl_noise':       0.1,
        'lr_actor':         3e-4,
        'lr_critic':         3e-4,
        'gamma':            0.995, 
        'weight_bc_reward': 10,
        'batch_size':       256,
        'process_count':    1,
        'eval_freq':        2e5,
        'eval_interp_freq': 1000,
        'tau':              0.005,
        'policy_noise':     0.2,
        'noise_clip':       0.5,
        'policy_freq':      20,
        'eval_episodes':    3, 
        'num_eval_env':     10,
        'max_episode_len':  500,  #
        'layer_N_critic':   2,
        'layer_N_actor':    2,
        'angle_loss_coeff':    10,
        'hidden_size':      400
    }),
}