# Hopper-v2 env
# two objectives
# running speed, jumping height

import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env
from os import path
import gym
import d4rl

class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self):
        self.obj_dim = 2
        mujoco_env.MujocoEnv.__init__(self, model_path = path.join(path.abspath(path.dirname(__file__)), "assets/hopper.xml"), frame_skip = 5)
        utils.EzPickle.__init__(self)
        self.action_space_type = "Continuous"
        self.reward_space = np.zeros((2,))

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        # if isinstance(a, (np.ndarray)):
        #     a = a[0]
        a = np.clip(a, [-2.0, -2.0, -4.0], [2.0, 2.0, 4.0])
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward_others = alive_bonus - 2e-4 * np.square(a).sum()
        reward_run = 1.5 * (posafter - posbefore) / self.dt + reward_others
        reward_jump = 12. * (height - self.init_qpos[1]) + reward_others
        s = self.state_vector()
        done = not((s[1] > 0.4) and abs(s[2]) < np.deg2rad(90) and abs(s[3]) < np.deg2rad(90) and abs(s[4]) < np.deg2rad(90) and abs(s[5]) < np.deg2rad(90))

        ob = self._get_obs()
        return ob, np.array([reward_run, reward_jump],dtype=np.float32), done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        c = 1e-3
        new_qpos = self.init_qpos + self.np_random.uniform(low=-c, high=c, size=self.model.nq)
        new_qpos[1] = self.init_qpos[1]
        new_qvel = self.init_qvel + self.np_random.uniform(low=-c, high=c, size=self.model.nv)
        self.set_state(new_qpos, new_qvel)
        # qpos = self.init_qpos + self.np_random.uniform(
        #     low=-0.005, high=0.005, size=self.model.nq
        # )
        # qvel = self.init_qvel + self.np_random.uniform(
        #     low=-0.005, high=0.005, size=self.model.nv
        # )
        # self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20

