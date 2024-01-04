from __future__ import absolute_import, division, print_function
import sys
# importing time module
import time
from datetime import datetime
import torch
import random
import torch.optim as optim
import torch.multiprocessing as mp
import os

sys.path.append('../')
from tqdm import tqdm
import lib

import lib.common_ptan as ptan

import numpy as np
import gym
import moenvs
from lib.utilities.MORL_utils import MOOfflineEnv
from lib.utilities.common_utils import make_config
from scipy.interpolate import RBFInterpolator

from collections import namedtuple, deque
import copy
import wandb
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", default='PR_dybc', help="PR_dybc, CQL", type=str)
    parser.add_argument("--policy_regularization", default='Diffusion-QL', help="TD3+BC, CVAE-QL, BCQ, BEAR, Diffusion-QL", type=str) 
    parser.add_argument("--env", default="Hopper")  
    parser.add_argument("--seed", default=0, type=int)  
    parser.add_argument("--use_wandb", default=1, type=int)   
    parser.add_argument("--test_only", default=0, type=int)   
    parser.add_argument("--load_model", default=0, type=int)   
    parser.add_argument("--dataset", default='d4rl', type=str)
    parser.add_argument("--num_objective", default=2, type=int)   
    parser.add_argument("--dataset_type", default=None)
    parser.add_argument("--policy_freq", default=2, type=int)
    parser.add_argument("--use_angleterm", default=0, type=int)
    parser.add_argument("--angle_loss_coeff", default=10.0, type=float)     
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--normalize_states", default=1, type=int)
    parser.add_argument("--pref_perturb_theta", default=1.0, type=float)
    parser.add_argument("--pref_gen_way", default='L1_return', type=str, help='L1_return, highest_return, None')

    parser.add_argument("--time_steps", default=1000000, type=int)
    parser.add_argument("--eval_freq", default=1000000, type=int)
    parser.add_argument("--weight_num", default=1, type=int)
    parser.add_argument("--w_bc_min", default=0.2, type=float)
    parser.add_argument("--w_bc_decay_last", default=0, type=int)
    parser.add_argument("--gamma", default=0.995, type=float)
    parser.add_argument("--weight_bc_reward", default=1.0, type=float)
    
    parser.add_argument("--reward_normalize", default=None, type=str)
    parser.add_argument("--eval_episodes", default=5, type=int) 
    parser.add_argument("--eval_way", default="exhaust|mc", help="exhaust, mc or tderror", type=str) 
    parser.add_argument("--mc_wbc_init_mean", default=0.0, type=float)  
    parser.add_argument("--mc_wbc_init_logstd", default=0.0, type=float)  
    parser.add_argument("--mc_sample_time", default=3, type=int)  
    parser.add_argument("--mc_sample_num", default=10, type=int)  
    parser.add_argument("--mc_update_num", default=20, type=int)  
    parser.add_argument("--num_eval_env", default=10, type=int)
    parser.add_argument("--w_step_size_final_eval", default=0.01, type=float)
    parser.add_argument("--w_step_size", default=0.05, type=float)
    parser.add_argument("--w_bc_size", default=0.05, type=float)
    parser.add_argument("--lr_decay", default=1, type=int)                           
    parser.add_argument("--policy_noise", default=0.2, type=float)
    parser.add_argument("--batch_size", default=256, type=int)
    
    parser.add_argument("--fixed_wbc", default=None, type=float)
    parser.add_argument("--fixed_pref1", default=None, type=float)
    parser.add_argument("--fixed_pref1_traj_num", default=None, type=int) 
    parser.add_argument("--pre_sample_traj_num", default=None, type=int)
    parser.add_argument("--dataset_preprocess", default=None, type=str, help="ret_scale, resampled_by_pref1, resampled_by_pref2") 
    
    parser.add_argument("--use_envelope_ql", default=0, type=int)  
    #cql paras
    parser.add_argument("--conservative_weight", default=1.0, type=float)
    #diffusion paras
    parser.add_argument("--diffusion_n_timesteps", default=5, type=int)

    start_time = time.time()
    input_args = parser.parse_args()
    USE_WANDB = input_args.use_wandb
    SEED = input_args.seed
    env_name = input_args.env
    dataset_type = input_args.dataset_type
    name = f"{env_name}_MOOF_TD3_HER"
    args = lib.utilities.settings.HYPERPARAMS[name]
    for arg in vars(input_args):
        setattr(args, arg, getattr(input_args, arg))
    name = f"{env_name}_{input_args.dataset}_{dataset_type}_{args.num_objective}obj"
    args.plot_name  = name
    args.name = name
    PROCESSES_COUNT = args.process_count

    torch.manual_seed(SEED)
    random.seed(SEED)
    np.random.seed(SEED)  
    
    os.environ['CUDA_VISIBLE_DEVICES'] = f"{args.gpu}"
    device = torch.device(f"cuda" if args.cuda else "cpu")
    args.device = device
    # setup the environment
    env_main = MOOfflineEnv(env_name, args.dataset, args.num_objective)
    test_env_main = [(lambda: MOOfflineEnv(env_name, args.dataset, args.num_objective)) for _ in range(args.num_eval_env)]
    env_main.seed(SEED)

    #Initialize environment related arguments
    args.obs_shape = env_main.observation_space.shape[0]
    args.action_shape = env_main.action_space.shape[0]
    args.reward_size = len(env_main.reward_space)+(args.algo=='PR_dybc')
    args.max_action = env_main.action_space.high
    if args.dataset=='d4rl':
        args.max_episode_len = 1000  
    elif args.dataset=='d4morl': 
        args.max_episode_len = 500
    else:
        args.max_episode_len = env_main._max_episode_steps
    
    if args.num_objective==2 and args.fixed_pref1 is not None and args.fixed_pref1<=1.0 and args.fixed_pref1>=0.0: 
        args.fixed_pref1 = np.array([args.fixed_pref1, 1.0-args.fixed_pref1])
    else:
        args.fixed_pref1 = None

    if args.num_objective==2 and args.fixed_wbc is not None and args.fixed_wbc<=1.0 and args.fixed_wbc>=0.0: 
        pass
        #args.eval_way = 'exhaust'
    else:
        args.fixed_wbc = None
          
    writer = lib.utilities.common_utils.WandbWriter(USE_WANDB, "offline multi-objective", args, SEED)
    args.writer = writer

    #Initialize the networks
    behavior_prior = None
    critic = lib.models.networks.Critic(args).to(device)
    if args.algo=='CQL':
        actor = lib.models.networks.Actor(args, is_gaussian=True).to(device)
    elif args.algo=='PR_dybc':
        if args.policy_regularization=='TD3+BC':
            actor = lib.models.networks.Actor(args, is_gaussian=False).to(device)
        elif args.policy_regularization=='CVAE-QL':
            actor = lib.models.networks.VAE(args, reward_size=args.reward_size).to(device)
        elif args.policy_regularization=='BEAR':
            from lib.diffusion.agents.diffusion import Diffusion
            from lib.diffusion.agents.model import MLP as Diffusion_MLP
            actor = lib.models.networks.Actor(args, is_gaussian=True).to(device)
            behavior_model = Diffusion_MLP(args.obs_shape, args.action_shape, 0, args.device)
            behavior_prior = Diffusion(behavior_model, args)
            behavior_prior.load_state_dict(torch.load(f'./bc_model/Diffusion/{env_name}_{dataset_type}.pth', map_location='cpu'))
            behavior_prior = behavior_prior.to(device)
        elif args.policy_regularization=='BCQ':
            actor = lib.models.networks.BCQ_Actor(args).to(device)
            behavior_prior = lib.models.networks.VAE(args, reward_size=args.reward_size-1).to(device)
        elif args.policy_regularization=='Diffusion-QL':
            from lib.diffusion.agents.diffusion import Diffusion
            from lib.diffusion.agents.model import MLP
            mlp_model = MLP(args.obs_shape, args.action_shape, args.reward_size, args.device).to(device)
            actor = Diffusion(mlp_model, args, n_timesteps=args.diffusion_n_timesteps).to(device)

    #Edit the neural network model name
    args.name_model = args.name
    
    #Load previously trained model
    if args.test_only:
        load_path = "Exps/{}/{}/".format(name, args.seed)
        model_actor = torch.load("{}{}.pkl".format(load_path,"{}_{}_{}".format(args.scenario_name, args.name_model,'final_actor'))) # Change the model name accordingly
        actor.load_state_dict(model_actor)
        model_critic = torch.load("{}{}.pkl".format(load_path,"{}_{}_{}".format(args.scenario_name, args.name_model,'final_critic')))  # Change the model name accordingly
        critic.load_state_dict(model_critic)
        print('load model!!!')

    #Initialize preference spaces
    w_batch_test = lib.utilities.MORL_utils.generate_w_batch_test(args, step_size = args.w_step_size_final_eval, reward_size=len(env_main.reward_space))
    w_batch_eval = lib.utilities.MORL_utils.generate_w_batch_test(args, step_size = args.w_step_size, reward_size=len(env_main.reward_space))

    if args.fixed_pref1 is not None:
        w_batch_test = np.array([args.fixed_pref1])
        w_batch_eval = np.array([args.fixed_pref1])

    #Initialize Experience Source and Replay Buffer
    replay_buffer_main = ptan.experience.ExperienceReplayBuffer_HER_MOOF(args)
    replay_buffer_main.load_from_dataset(env_main, dataset_type=dataset_type)
    if args.normalize_states:
        mean,std = replay_buffer_main.normalize_states()  
    else:
        mean, std = 0, 1
    state_normalizer = lambda state: torch.tensor(np.array((state-mean)/std, dtype=np.float32)) 

    agent_main = ptan.agent.MOOF_TD3_HER(actor,critic, device, behavior_prior, args, state_normalizer)
    interp = lambda x: x
    agent_main.interp = interp

    time_step = 0
    if not args.test_only:
        # Main Loop
        done_episodes = 0
        time_step = 0
        eval_cnt = 1
        eval_cnt_ep = 1
        for ts in tqdm(range(0, args.time_steps), mininterval=10): #iterate through the fixed number of timesteps
            
            # Learn from the minibatch
            agent_main.learn(replay_buffer_main, writer) 

            time_step = ts 
            # Evaluate agent
            if ts > args.eval_freq*eval_cnt:
                test_time_start = time.time()
                eval_cnt +=1
                print(f"Time steps: {time_step}, Episode Count of Each Process: {time_step}")

                if args.algo=='PR_dybc':
                    if "mc" in args.eval_way:
                        hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_by_mc_adaptive(test_env_main, env_main, agent_main, w_batch_eval, args, time_step, eval_episodes=args.eval_episodes)
                        lib.utilities.MORL_utils.store_results( [], hypervolume, sparsity, time_step, writer, args)
                        lib.utilities.MORL_utils.plot_offline_objs(args,objs,name,ext=f'{time_step}_mc')
                    if "exhaust" in args.eval_way:
                        hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_by_exhaust_wbc(test_env_main, env_main, agent_main, w_batch_eval, args, time_step, eval_episodes=args.eval_episodes)
                        lib.utilities.MORL_utils.store_results( [], hypervolume, sparsity, time_step, writer, args)
                        lib.utilities.MORL_utils.plot_offline_objs(args,objs,name,ext=f'{time_step}_exhaust')
                else:
                    hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_agent(test_env_main, env_main, agent_main, w_batch_eval, args, time_step, eval_episodes=args.eval_episodes)
                    lib.utilities.MORL_utils.store_results( [], hypervolume, sparsity, time_step, writer, args)
                    lib.utilities.MORL_utils.plot_offline_objs(args,objs,name,ext=f'{time_step}_eval')

                lib.utilities.common_utils.save_model(actor, args, name = name, ext ='actor_{}'.format(time_step))
                lib.utilities.common_utils.save_model(critic, args, name = name,ext ='critic_{}'.format(time_step))
                print(f"Eval time: {time.time()-test_time_start}")
        
        print(f"Total Number of Time Steps: {time_step}")
        lib.utilities.common_utils.save_model(actor, args, name = name,ext ='final_actor')
        lib.utilities.common_utils.save_model(critic, args, name = name,ext ='final_critic')

    
    #Evaluate the final agent
    if args.algo=='PR_dybc':
        if "mc" in args.eval_way:
            hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_by_mc_adaptive(test_env_main, env_main, agent_main, w_batch_test, args, time_step, eval_episodes=args.eval_episodes)
            lib.utilities.MORL_utils.store_results( [], hypervolume, sparsity, time_step, writer, args)
            lib.utilities.MORL_utils.plot_offline_objs(args,objs,name,ext=f'final_mc' if args.fixed_wbc is None else f'final_mc_fixbc{args.fixed_wbc}')
        if "exhaust" in args.eval_way:
            hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_by_exhaust_wbc(test_env_main, env_main, agent_main, w_batch_test, args, time_step, eval_episodes=args.eval_episodes)
            lib.utilities.MORL_utils.store_results( [], hypervolume, sparsity, time_step, writer, args)
            lib.utilities.MORL_utils.plot_offline_objs(args,objs,name,ext=f'final_exhaust' if args.fixed_wbc is None else f'final_exhaust_fixbc{args.fixed_wbc}')
    else:
        hypervolume, sparsity, objs = lib.utilities.MORL_utils.eval_agent(test_env_main, env_main, agent_main, w_batch_test, args, time_step, eval_episodes=args.eval_episodes)
        lib.utilities.MORL_utils.store_results( [], hypervolume, sparsity, time_step, writer, args)
        lib.utilities.MORL_utils.plot_offline_objs(args,objs,name,ext=f'final_eval')
    
    print("Time Consumed")
    print("%0.2f minutes" % ((time.time() - start_time)/60))
    wandb.finish()
    

    
