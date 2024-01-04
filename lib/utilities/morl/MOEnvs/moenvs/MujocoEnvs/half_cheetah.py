# HalfCheetah-v2 env
# two objectives
# running speed, energy efficiency

import d4rl
import gym
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from gym import Wrapper
from os import path

class HalfCheetahEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 2
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/half_cheetah.xml"), frame_skip = 5)
        utils.EzPickle.__init__(self)
        self.action_space_type = "Continuous"
        self.reward_space = np.zeros((2,))
    
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        # if isinstance(action, (np.ndarray)):
        #     action = action[0]
        action = np.clip(action, -1.0, 1.0)
        self.do_simulation(action, self.frame_skip)
        xposafter, ang = self.sim.data.qpos[0], self.sim.data.qpos[2]
        ob = self._get_obs()
        alive_bonus = 1.0

        reward_run = (xposafter - xposbefore)/self.dt
        reward_run = min(4.0, reward_run) + alive_bonus
        reward_energy = 4.0 - 1.0 * np.square(action).sum() + alive_bonus

        done = not (abs(ang) < np.deg2rad(50))

        return ob, np.array([reward_run, reward_energy],dtype=np.float32), done, {}
    
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])
    
    def reset_model(self):
        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=-0.1, high=0.1, size=self.model.nq
        # )
        # qvel = self.init_qvel + self.np_random.standard_normal(self.model.nv) * 0.1
        # self.set_state(qpos, qvel)
        c = 1e-3
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq),
            self.init_qvel + c * self.np_random.standard_normal(self.model.nv)
        )
        return self._get_obs()
    
    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5


