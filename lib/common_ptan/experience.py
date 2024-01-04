import gym
import torch
import numpy as np
import d4rl
import pickle

class ExperienceReplayBuffer_HER_MOOF(object):
    def __init__(self, args, max_size=int(2e6), ):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        self.device = args.device
        self.args = args


    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample(self, batch_size, w_bc_min=None):
        ind = np.random.randint(0, self.size, size=batch_size)
        rewards = np.concatenate([self.reward[ind], np.zeros((batch_size, 1))], axis=1)
        weight_num = self.args.weight_num

        if w_bc_min is None:
            w_bc_min = self.args.w_bc_min
        
        ori_preference = self.ori_preference[ind]
        obj_preference = self.gen_random_perturbed_preference(ori_preference, weight_num, w_bc_min, False)

        w_bc = np.random.uniform(w_bc_min, 1, size=(len(obj_preference), 1))
        if self.args.fixed_wbc is not None:
            w_bc[:, :] = self.args.fixed_wbc
        preference = np.concatenate([obj_preference * (1-w_bc), w_bc], axis=1)
            
        return (
            torch.FloatTensor(self.state[ind].repeat(weight_num, axis=0)).to(self.device),
            torch.FloatTensor(self.action[ind].repeat(weight_num, axis=0)).to(self.device),
            torch.FloatTensor(self.next_state[ind].repeat(weight_num, axis=0)).to(self.device),
            torch.FloatTensor(rewards.repeat(weight_num, axis=0)).to(self.device),
            torch.FloatTensor(self.not_done[ind].repeat(weight_num, axis=0)).to(self.device),
            torch.FloatTensor(obj_preference).to(self.device),
            torch.FloatTensor(preference).to(self.device),
        )

    def cal_vec_cos(self, vec1, vec2):
        assert vec1.shape==vec2.shape
        return np.sum(vec1*vec2, axis=-1) / (np.linalg.norm(vec1, axis=-1)*np.linalg.norm(vec2, axis=-1))
        
    def gen_random_preference(self, size, w_bc_min):
        w_batch_rnd = np.random.randn(size, self.args.reward_size-(self.args.algo=='PR_dybc'))
        w_bc = np.random.uniform(w_bc_min, 1, size=(size, 1))
        w_batch_obj = np.abs(w_batch_rnd) / np.linalg.norm(w_batch_rnd, ord=1, axis=1, keepdims=True)
        w_batch = np.concatenate([w_batch_obj * (1-w_bc), w_bc], axis=1)
        return w_batch_obj, w_batch
    
    def rotate_vectors(self, v1, v2, shrink_ratio):

        if np.abs(shrink_ratio-1)<0.001:
            return v2 / np.linalg.norm(v2, ord=1, axis=1, keepdims=True)
        
        if np.abs(shrink_ratio)<0.001:
            return v1 / np.linalg.norm(v1, ord=1, axis=1, keepdims=True)

        v1 = v1 / np.linalg.norm(v1, ord=2, axis=1, keepdims=True)
        vec_cos_old = self.cal_vec_cos(v1, v2)
        vec_cos_new = np.cos(shrink_ratio*np.arccos(np.clip(vec_cos_old, -0.99999, 0.99999)))
        
        vec_cos_new_square = np.square(vec_cos_new)
        v1v2 = np.einsum("ij,ij->i", v1, v2)
        v1v1 = np.einsum("ij,ij->i", v1, v1)
        v2v2 = np.einsum("ij,ij->i", v2, v2)
        a=np.square(v1v2)-v2v2*vec_cos_new_square
        b=2*v1v2*v1v1-2*v1v2*vec_cos_new_square
        c=np.square(v1v1)-v1v1*vec_cos_new_square
        discriminant = np.maximum(b**2 - 4*a*c, 0)

        root1 = (-b + np.sqrt(discriminant)) / (2*a+1e-8)
        root2 = (-b - np.sqrt(discriminant)) / (2*a+1e-8)
        root = np.maximum(root1, root2).reshape(-1, 1)
        
        new_vec = v1 + root * v2
        zero_vec_idx = np.all(np.abs(new_vec)<1e-4, 1)
        new_vec[zero_vec_idx] = v1[zero_vec_idx]
        new_vec = new_vec / np.linalg.norm(new_vec, ord=1, axis=1, keepdims=True)
        return new_vec
    
    def gen_random_perturbed_preference(self, ori_prefs, weight_num, w_bc_min, first_no_random=False):
        size = len(ori_prefs) * weight_num
        w_batch_rnd = np.random.randn(size, self.args.reward_size-(self.args.algo=='PR_dybc'))
        w_batch_obj = w_batch_rnd / np.linalg.norm(w_batch_rnd, ord=1, axis=1, keepdims=True)
        w_batch_obj = self.rotate_vectors(ori_prefs, w_batch_obj, self.args.pref_perturb_theta)
        w_batch_obj = np.abs(w_batch_obj) / np.linalg.norm(w_batch_obj, ord=1, axis=1, keepdims=True) 

        if self.args.fixed_pref1 is not None:
            w_batch_obj[:] = self.args.fixed_pref1.reshape(1, -1)
        return w_batch_obj 


    
    def preprocess_dataset(self, trajectories, env):
        if self.args.pre_sample_traj_num:
            rdm = np.random.RandomState(0)
            traj_idx = rdm.choice(len(trajectories), self.args.pre_sample_traj_num, replace=False)
            trajectories = [trajectories[idx] for idx in traj_idx]

        if self.args.pref_gen_way=='L1_return': #substitute the real behavior preferences
            for traj in trajectories:
                ret = np.sum(traj['raw_rewards'], axis=0, keepdims=True)
                ret = ret / np.sum(np.abs(ret))
                traj['preference'][:] = ret

        if self.args.fixed_pref1 is not None and self.args.fixed_pref1_traj_num:
            rank = np.array([self.cal_vec_cos(self.args.fixed_pref1, traj['preference'][0]) for traj in trajectories])
            traj_idx = np.argsort(-rank) 
            trajectories = [trajectories[idx] for idx in traj_idx[:self.args.fixed_pref1_traj_num]]

        for i in range(len(trajectories)):
            if env.spec.id=='MO-Hopper-v2':
                trajectories[i]['actions'] /= np.array([[2, 2 ,4]]) 
        return trajectories

    def load_from_dataset(self, env, dataset_type="amateur_uniform"):
        max_episode_len = self.args.max_episode_len
        trajectories = env.get_dataset(dataset_type)
        trajectories = self.preprocess_dataset(trajectories, env)

        self.size = np.sum([len(traj['observations']) for traj in trajectories])
        self.state = np.zeros((self.size, len(trajectories[0]['observations'][0])))
        self.next_state = np.zeros((self.size, len(trajectories[0]['observations'][0])))
        self.action = np.zeros((self.size, len(trajectories[0]['actions'][0])))
        self.reward = np.zeros((self.size, len(trajectories[0]['raw_rewards'][0])))
        self.ori_preference = np.zeros((self.size, len(trajectories[0]['preference'][0])))
        self.not_done = np.ones((self.size, 1), dtype=bool)
        returns = []
        self.pref_ret_pairs = []
        gamma = self.args.gamma**np.expand_dims(np.arange(max_episode_len), 1)
        cnt = 0
        for traj in trajectories:
            traj_len = len(traj['observations'])
            self.state[cnt:cnt+traj_len] = traj['observations']
            self.next_state[cnt:cnt+traj_len] = traj['next_observations']
            self.action[cnt:cnt+traj_len] = traj['actions']
            self.reward[cnt:cnt+traj_len] = traj['raw_rewards']
            self.ori_preference[cnt:cnt+traj_len] = traj['preference']
            self.not_done[cnt+traj_len-1] = (traj_len==max_episode_len)
            returns.append(np.sum(self.reward[cnt:cnt+traj_len]*gamma[:traj_len], 0))
            self.pref_ret_pairs.append((traj['preference'][0], np.sum(self.reward[cnt:cnt+traj_len], 0)))
            cnt += traj_len

        min_pref, max_pref = np.min(self.ori_preference, axis=0), np.max(self.ori_preference, axis=0)
        self.min_pref, self.max_pref = min_pref, max_pref
        cnt = 0
        self.v_min, self.v_max = np.min(returns, 0, keepdims=True), np.max(returns, 0, keepdims=True)
        self.v_mean, self.v_std = np.mean(returns, 0, keepdims=True), np.std(returns, 0, keepdims=True)
        self.reward = self.reward_prepreprocess(self.reward)

        print("num of original trajectory: ", len(trajectories))
        print("original pref range: ", min_pref, max_pref)
        print('episode_num:', len(returns))
        print('min_max:', self.v_min, self.v_max)
        print('mean, std:', self.v_mean, self.v_std)

    def reward_prepreprocess(self, reward):
        L, GAMMA = self.args.max_episode_len, self.args.gamma
        if self.args.reward_normalize == 'min_max': 
            reward = (reward - self.v_min*(1-GAMMA)/(1-GAMMA**L)) / (self.v_max - self.v_min + 1e-6)
        elif self.args.reward_normalize == 'z_score':
            reward = (reward - self.v_mean*(1-GAMMA)/(1-GAMMA**L)) / (self.v_std + 1e-6)
        elif self.args.reward_normalize == 'div_mean':
            reward = reward/ (np.abs(self.v_mean) + 1e-6)
        elif self.args.reward_normalize == 'div_max':
            reward = reward / (self.v_max + 1e-6)
        else:
            pass
        return reward

    def normalize_states(self, eps = 1e-3):
        mean = self.state.mean(0,keepdims=True)
        std = self.state.std(0,keepdims=True) + eps
        self.state = (self.state - mean)/std
        self.next_state = (self.next_state - mean)/std
        return mean, std

