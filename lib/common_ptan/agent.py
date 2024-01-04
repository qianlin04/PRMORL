"""
Agent is something which converts states into actions and has state
"""

import copy

import numpy as np
import torch
import random
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from . import actions

from torch.autograd import Variable
import collections
from torch.distributions import Normal


def default_states_preprocessor(states):
    """
    Convert list of states into the form suitable for model.
    :param states: list of numpy arrays with states
    :return: Variable
    """
    if len(states) == 1:
        np_states = np.expand_dims(states[0], 0)
    else:
        np_states = np.array([np.array(s, copy=False) for s in states], copy=False)
    return torch.tensor(np_states)


def float32_preprocessor(states):
    np_states = np.array(states, dtype=np.float32)
    return torch.tensor(np_states)  


class MOOF_TD3_HER:

    def __init__(self, actor, critic, device, behavior_prior, args, preprocessor=float32_preprocessor):
        
        super().__init__()

        self.args = args
        self.preprocessor = preprocessor
        self.device = device
        self.actor = actor
        self.critic = critic
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)        
        self.behavior_prior = behavior_prior
        self.behavior_optimizer = torch.optim.Adam(behavior_prior.parameters()) if behavior_prior is not None else None

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.args.lr_actor)
        if critic:
            self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.args.lr_critic)

        if self.args.lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=self.args.time_steps//1000, eta_min=0.)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=self.args.time_steps//1000, eta_min=0.)

        self.state_size = args.obs_shape
        self.action_size =  args.action_shape
        self.reward_size =  args.reward_size
        self.preference = None
        self.total_it = 0
        self.args = args
        self.w_ep = None
        self.weight_num = args.weight_num
        self.max_action = args.max_action[0]
        self.gamma = args.gamma
        self.tau  = args.tau
        self.policy_noise = args.policy_noise * self.max_action
        self.noise_clip = args.noise_clip * self.max_action
        self.expl_noise =args.expl_noise
        self.policy_freq = args.policy_freq
        self.deterministic = False
        self.w_batch = []
        self.interp = []

    def sample_actions(self, state, pref, actor=None, deterministic = False, need_log_pi = False):
        assert not need_log_pi or not deterministic
        if actor is None:
            actor = self.actor

        if self.args.algo=='CQL':
            mean, log_std = actor(state, pref)
            if deterministic:
                return torch.tanh(mean) * self.max_action
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            actions = y_t * self.max_action 
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.max_action
            if need_log_pi:
                return actions, log_prob
            else:
                return actions    
        elif self.args.policy_regularization=='Diffusion-QL':
            return actor(state, pref)
        elif self.args.policy_regularization=='CVAE-QL':
            return actor.decode(state, pref)
        elif self.args.policy_regularization=='BCQ':
            w_bc = pref[:, -1:]
            obj_pref = pref[:, :-1] / (1-w_bc+1e-8) 
            actions = self.behavior_prior.decode(state, obj_pref, )
            return actor(state, pref, actions, w_bc)
        elif self.args.policy_regularization=='TD3+BC':
            actions = actor(state, pref)
            if deterministic:
                return actions
            noise = torch.randn_like(actions).to(self.device)
            noise_clip = (noise*self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            actions = (actions + noise_clip).clamp(-self.max_action,self.max_action)
            if need_log_pi:
                normal = Normal(torch.zeros_like(actions).to(self.device), torch.ones_like(actions).to(self.device))
                log_prob = normal.log_prob(noise)
                return actions, log_prob
            else:
                return actions
            
            

    def __call__(self, states, preference, deterministic = False):
        # Set type for states
        if self.preprocessor is not None:
            states = self.preprocessor(states)
            if torch.is_tensor(states):
                states = states.to(self.device)
       
        # Choose action for a given policy 
        if self.args.algo=='PR_dybc' and self.args.policy_regularization=='Diffusion-QL':
            batch_size = states.shape[0]
            repeat_sample_num = 50
            state_rpt = torch.repeat_interleave(states, repeats=repeat_sample_num, dim=0)
            pref_rpt = torch.repeat_interleave(preference, repeats=repeat_sample_num, dim=0)
            with torch.no_grad():
                actions = self.actor_target.sample(state_rpt, pref_rpt)
                q1, q2 = self.critic_target(state_rpt, pref_rpt, actions)
                q1 = torch.bmm(pref_rpt[:,:-1].unsqueeze(1), q1[:,:-1].unsqueeze(2)).squeeze()
                q2 = torch.bmm(pref_rpt[:,:-1].unsqueeze(1), q2[:,:-1].unsqueeze(2)).squeeze()
                q_value = torch.min(q1, q2)
                q_value = q_value.reshape(batch_size, repeat_sample_num)
                idx = torch.argmax(q_value, dim=1, keepdim=True)
                idx = idx.unsqueeze(-1).expand(-1, -1, self.action_size)
                actions = actions.reshape(batch_size, repeat_sample_num, self.action_size)
                actions = torch.gather(actions, dim=1, index=idx)
            actions = actions.squeeze(1).cpu().data.numpy()
        else:
            with torch.no_grad():
                actions = self.sample_actions(states, preference, self.actor, deterministic=deterministic).cpu().data.numpy()
        return actions
    
    # MMD functions
    def compute_gau_kernel(self, x, y, sigma):
        batch_size = x.shape[0]
        x_size = x.shape[1]
        y_size = y.shape[1]
        dim = x.shape[2]
        tiled_x = x.view(batch_size, x_size, 1, dim).repeat([1, 1, y_size, 1])
        tiled_y = y.view(batch_size, 1, y_size, dim).repeat([1, x_size, 1, 1])
        return torch.exp(-(tiled_x - tiled_y).pow(2).sum(dim=3) / (2 * sigma))

    # MMD functions
    def compute_lap_kernel(self, x, y, sigma):
        batch_size = x.shape[0]
        x_size = x.shape[1]
        y_size = y.shape[1]
        dim = x.shape[2]
        tiled_x = x.view(batch_size, x_size, 1, dim).repeat([1, 1, y_size, 1])
        tiled_y = y.view(batch_size, 1, y_size, dim).repeat([1, x_size, 1, 1])
        return torch.exp(-torch.abs(tiled_x - tiled_y).sum(dim=3) / sigma)

    def compute_mmd(self, x, y, kernel='lap'):
        if kernel == 'gau':
            x_kernel = self.compute_gau_kernel(x, x, 20)
            y_kernel = self.compute_gau_kernel(y, y, 20)
            xy_kernel = self.compute_gau_kernel(x, y, 20)
        else:
            x_kernel = self.compute_lap_kernel(x, x, 10)
            y_kernel = self.compute_lap_kernel(y, y, 10)
            xy_kernel = self.compute_lap_kernel(x, y, 10)
        square_mmd = x_kernel.mean((1, 2)) + y_kernel.mean((1, 2)) - 2 * xy_kernel.mean((1, 2))
        return square_mmd

    def cal_bc_reward(self, policy_action, behavior_action, state, pref):
        if self.args.policy_regularization=='Diffusion-QL':
            bc_reward = -self.args.weight_bc_reward * self.actor.loss(behavior_action, state, pref)
        elif self.args.policy_regularization=='CVAE-QL':
            recon, mean, std = self.actor(state, pref, behavior_action)
            recon_loss = F.mse_loss(recon, behavior_action)
            KL_loss    = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            bc_reward = -self.args.weight_bc_reward * (recon_loss + 0.5 * KL_loss)
        elif self.args.policy_regularization=='TD3+BC':
            bc_reward = -self.args.weight_bc_reward * torch.square(policy_action-behavior_action).mean(-1)
        elif self.args.policy_regularization=='BEAR':
            assert self.behavior_prior is not None
            #if self.actor.is_gaussian:
            m, n = 10, 10
            state_rp_m = state.repeat_interleave(m, 0)
            state_rp_n = state.repeat_interleave(n, 0)
            pref_rp_m = pref.repeat_interleave(m, 0)
            with torch.no_grad():
                prior_a = self.behavior_prior(state_rp_n).view(len(state), n, -1)
            a_rep = self.sample_actions(state_rp_m, pref_rp_m, self.actor).view(len(state), m, -1)
            bc_reward = -self.args.weight_bc_reward * self.compute_mmd(prior_a, a_rep)
            # else:
            #     batch_size = self.args.batch_size
            #     weight_num = self.args.weight_num
            #     with torch.no_grad():
            #         prior_a = self.behavior_prior(state).view(batch_size, weight_num, -1)
            #     a_rep = policy_action.view(batch_size, weight_num, -1)
            #     bc_reward = -self.compute_mmd(prior_a, a_rep).repeat_interleave(weight_num, 0)
        else:
            bc_reward = torch.zeros(len(policy_action)).to(self.device)
        return bc_reward
    
    def cal_conservative_loss(self, states, actions, next_states, w_batch_input, w_obj_batch, q1_pred, q2_pred):
        ## add CQL
        if self.actor.is_gaussian:
            num_random = 10
            num_sample = len(states)
            states = states.repeat_interleave(num_random, 0)
            next_states = next_states.repeat_interleave(num_random, 0)
            w_batch_input = w_batch_input.repeat_interleave(num_random, 0)
            w_obj_batch = w_obj_batch.repeat_interleave(num_random, 0)
        else:
            num_sample = self.args.batch_size
            num_random = self.args.weight_num

        random_actions_tensor = torch.FloatTensor(len(states), actions.shape[-1]).uniform_(-1, 1).to(self.device) 
        with torch.no_grad():
            if self.actor.is_gaussian:
                curr_actions_tensor, curr_log_pis = self.sample_actions(states, w_batch_input, self.actor, need_log_pi=True)
                new_curr_actions_tensor, new_log_pis = self.sample_actions(next_states, w_batch_input, self.actor, need_log_pi=True)
                curr_log_pis, new_log_pis = curr_log_pis.view(num_sample, num_random).detach(), new_log_pis.view(num_sample, num_random).detach()
                random_density = np.log(0.5 ** curr_actions_tensor.shape[-1])
            else:
                curr_actions_tensor, curr_log_pis = self.sample_actions(states, w_batch_input, self.actor), 0
                new_curr_actions_tensor, new_log_pis = self.sample_actions(next_states, w_batch_input, self.actor), 0
                random_density = 0
        
        q1_rand, q2_rand = self.critic(states, w_batch_input, random_actions_tensor)
        q1_curr_actions, q2_curr_actions = self.critic(states, w_batch_input, curr_actions_tensor)
        q1_next_actions, q2_next_actions = self.critic(states, w_batch_input, new_curr_actions_tensor)
        
        q1_rand = torch.bmm(w_obj_batch.unsqueeze(1), q1_rand.unsqueeze(2)).squeeze().reshape(num_sample, num_random)
        q2_rand = torch.bmm(w_obj_batch.unsqueeze(1), q2_rand.unsqueeze(2)).squeeze().reshape(num_sample, num_random)
        q1_curr_actions = torch.bmm(w_obj_batch.unsqueeze(1), q1_curr_actions[:,:].unsqueeze(2)).squeeze().reshape(num_sample, num_random)
        q2_curr_actions = torch.bmm(w_obj_batch.unsqueeze(1), q2_curr_actions[:,:].unsqueeze(2)).squeeze().reshape(num_sample, num_random)
        q1_next_actions = torch.bmm(w_obj_batch.unsqueeze(1), q1_next_actions[:,:].unsqueeze(2)).squeeze().reshape(num_sample, num_random)
        q2_next_actions = torch.bmm(w_obj_batch.unsqueeze(1), q2_next_actions[:,:].unsqueeze(2)).squeeze().reshape(num_sample, num_random)

        cat_q1 = torch.cat([q1_rand-random_density, q1_next_actions-new_log_pis, q1_curr_actions-curr_log_pis], 1)
        cat_q2 = torch.cat([q2_rand-random_density, q2_next_actions-new_log_pis, q2_curr_actions-curr_log_pis], 1)
            
        min_qf1_loss = torch.logsumexp(cat_q1, dim=1,).mean() 
        min_qf2_loss = torch.logsumexp(cat_q2, dim=1,).mean() 
        min_qf1_loss = (min_qf1_loss - q1_pred.mean()) * self.args.conservative_weight
        min_qf2_loss = (min_qf2_loss - q2_pred.mean()) * self.args.conservative_weight
        return min_qf1_loss+min_qf2_loss

    # Learn from batch
    def learn(self, replay_buffer, writer):
        self.writer = writer
        FloatTensor = torch.cuda.FloatTensor if self.args.cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if self.args.cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if self.args.cuda else torch.ByteTensor
        weight_num = self.args.weight_num
        batch_size = self.args.batch_size

        self.total_it += 1

        if self.args.w_bc_decay_last:
            w_bc_min = 1.0 - (1.0 - self.args.w_bc_min) * min(self.total_it/(self.args.w_bc_decay_last+1), 1.0) 
        else:
            w_bc_min = self.args.w_bc_min

        batch = replay_buffer.sample(batch_size, w_bc_min) 
        state_batch, action_batch, next_state_batch,\
              reward_batch, not_done, w_obj_batch, w_batch = batch

        w_batch_np_critic = w_obj_batch
        w_batch_np_actor = w_obj_batch
        if self.args.algo=='PR_dybc':
            w_batch_input = w_batch.clone()
        else:
            w_batch_input = w_obj_batch

        if self.args.policy_regularization=='BCQ':
            recon, mean, std = self.behavior_prior(state_batch, w_obj_batch, action_batch)
            recon_loss = F.mse_loss(recon, action_batch)
            KL_loss    = -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
            vae_loss = recon_loss + 0.5 * KL_loss

            self.behavior_optimizer.zero_grad()
            vae_loss.backward()
            self.behavior_optimizer.step()
 
        with torch.no_grad():
                       
            # Compute the target Q value
            noise_next_action_batch = self.sample_actions(next_state_batch, w_batch_input, self.actor_target, deterministic = False)
            target_Q1, target_Q2 = self.critic_target(next_state_batch, w_batch_input, noise_next_action_batch)

            if self.args.use_envelope_ql:
                target_Q_min = torch.min(target_Q1, target_Q2).reshape(batch_size, weight_num, -1)
                target_Q_min_tp = target_Q_min.transpose(1,2)
                wTauQ = torch.bmm(w_obj_batch.reshape(batch_size, weight_num, -1), target_Q_min_tp[:,:w_obj_batch.shape[-1],:])
                wTauQ_envelope, wTauQ_envelope_idx = torch.max(wTauQ, dim=2) 
                select_idx = wTauQ_envelope_idx.unsqueeze(-1).expand(-1, -1, target_Q_min.shape[-1])
                target_Q = target_Q_min.gather(dim=1, index=select_idx).reshape(batch_size*weight_num, -1)
            else:
                wTauQ1 = torch.bmm(w_batch_np_critic.unsqueeze(1),target_Q1[:,:w_batch_np_critic.shape[-1]].unsqueeze(2)).squeeze()
                wTauQ2 = torch.bmm(w_batch_np_critic.unsqueeze(1),target_Q2[:,:w_batch_np_critic.shape[-1]].unsqueeze(2)).squeeze()
                _, wTauQ_min_idx = torch.min(torch.cat((wTauQ1.unsqueeze(-1),wTauQ2.unsqueeze(-1)),dim=-1),1,keepdim=True)
                wTauQ_min_idx = wTauQ_min_idx.unsqueeze(-1).expand(-1, -1, target_Q1.shape[1])
                target_Q_cat = torch.cat((target_Q1.unsqueeze(1),target_Q2.unsqueeze(1)),dim=1)
                target_Q = target_Q_cat.gather(1, wTauQ_min_idx).squeeze(1)

            target_Q = not_done * self.gamma * target_Q
            target_Q += reward_batch[:, :target_Q.shape[-1]]
        
        # Get current Q values
        current_Q1, current_Q2 = self.critic(state_batch, w_batch_input, action_batch)
        critic_loss1 = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        
        critic_loss = critic_loss1
        if self.args.use_angleterm:
            angle_term_1 = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_np_critic,(current_Q1)[:, :w_batch_np_critic.shape[-1]]),0, 0.9999)))
            angle_term_2 = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_np_critic,(current_Q2)[:, :w_batch_np_critic.shape[-1]]),0, 0.9999)))
            angle_loss = angle_term_1.mean() + angle_term_2.mean()
            critic_loss += angle_loss
        if self.args.algo=='CQL':
            conservative_loss = self.cal_conservative_loss(state_batch, action_batch, next_state_batch, w_batch_input, w_obj_batch, current_Q1, current_Q2)
            critic_loss += conservative_loss
        
        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5 if self.args.policy_regularization=='Diffusion-QL' else 100) 
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute Actor Loss
            policy_action = self.sample_actions(state_batch, w_batch_input, actor=self.actor, deterministic=(self.args.policy_regularization=='TD3+BC'))
            Q1, Q2 = self.critic(state_batch, w_batch_input, policy_action)    
            Q = Q1 if np.random.uniform() > 0.5 else Q2
            if self.args.algo=='PR_dybc':
                bc_term = self.cal_bc_reward(policy_action, action_batch, state_batch, w_batch_input)
                Q[:, -1] = bc_term
            wQ = torch.bmm(w_obj_batch.unsqueeze(1), Q[:,:w_obj_batch.shape[-1]].unsqueeze(2)).squeeze()
            lmbda = 1/wQ.abs().mean().detach()

            Q[:, :w_batch_np_actor.shape[-1]] = Q[:, :w_batch_np_actor.shape[-1]] * lmbda 
            if self.args.algo=='PR_dybc':
                if self.args.policy_regularization!='BCQ':
                    wQ = torch.bmm(w_batch.unsqueeze(1), Q.unsqueeze(2)).squeeze()
                else:
                    wQ = torch.bmm(w_batch_np_actor.unsqueeze(1), Q[:, :w_batch_np_actor.shape[-1]].unsqueeze(2)).squeeze()
            else:
                wQ = torch.bmm(w_batch_np_actor.unsqueeze(1), Q.unsqueeze(2)).squeeze()
        
            if self.args.use_angleterm:
                angle_term = torch.rad2deg(torch.acos(torch.clamp(F.cosine_similarity(w_batch_np_actor, Q[:,:w_batch_np_actor.shape[-1]]),0, 0.9999)))
                actor_loss = -wQ.mean() + self.args.angle_loss_coeff*angle_term.mean()
            else:
                actor_loss = -wQ.mean()
            
            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5 if self.args.policy_regularization=='Diffusion-QL' else 100)  
            self.actor_optimizer.step()
                       
            # Soft update the target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            actor_update_step = self.total_it // self.policy_freq
            if self.args.policy_regularization!='Diffusion-QL' or \
                actor_update_step % 5==0 and self.total_it>=1000:
                for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                        
            # Write the results to tensorboard
            if (self.total_it % 5000) == 0:
                w_batch_np_actor = torch.tensor(w_batch_np_actor).type(FloatTensor).to(self.device)
                scale_q = torch.bmm(w_batch_np_actor.unsqueeze(1), Q[:,:w_batch_np_actor.shape[-1]].unsqueeze(2)).squeeze().detach().cpu().numpy()
                writer.log_table('prefs vs scale_q', prefs=w_batch_np_actor.detach().cpu().numpy()[:, 0], scale_q=scale_q)
                writer.add_scalar('Loss/Actor_Loss'.format(), actor_loss, self.total_it)
                writer.add_scalar('Loss/Actor_wq'.format(), wQ.mean(), self.total_it)
                writer.add_scalar('Loss/Critic_Loss'.format(), critic_loss1, self.total_it)
                if self.args.algo=='PR_dybc':
                    writer.add_scalar('Loss/bc_term'.format(), bc_term.mean(), self.total_it)
                if self.args.use_angleterm:
                    writer.add_scalar('Loss/Actor_Loss_angleterm'.format(), angle_term.mean(), self.total_it)
                    writer.add_scalar('Loss/Critic_Loss_angleterm'.format(), angle_term_1.mean(), self.total_it)
                if self.args.algo=='CQL':
                    writer.add_scalar('Loss/Conservative_Loss'.format(), conservative_loss, self.total_it)
                    
                for k in range(Q.shape[1]):
                    writer.add_scalar(f'Loss/Critic_objective_{k}', target_Q1.reshape(batch_size, weight_num, -1)[:, :, k].mean(), self.total_it)
                    writer.add_scalar(f'Loss/Actor_objective_{k}', Q[:, k].mean(), self.total_it)
        
        if self.args.lr_decay and self.total_it % 1000==0: 
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

    def reset_preference(self):
        self.w_ep = None           




