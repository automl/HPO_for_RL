from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
import os
import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env


class HopperEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self,
                 xml_file='hopper.xml'):

        utils.EzPickle.__init__(**locals())
        dir_path = os.path.dirname(os.path.realpath(__file__))
        self.prev_qpos = None
        mujoco_env.MujocoEnv.__init__(self, '%s/assets/hopper.xml' % dir_path, 4)
        

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.prev_qpos = np.copy(self.sim.data.qpos.flat)
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        
        s = self.state_vector()

        alive = (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))

        alive_bonus = 1.0 if alive else 0.0

        forward_reward = (posafter - posbefore) / self.dt
        control_cost = 1e-3 * np.square(a).sum()
        reward = forward_reward - control_cost + alive_bonus

        done = False
        ob = self._get_obs()
        return ob, reward, done, {"forward_reward" : forward_reward}

    def _get_obs(self):

        position = self.sim.data.qpos.flat.copy()
        velocity = self.sim.data.qvel.flat.copy()

        return np.concatenate([
            (position[:1] - self.prev_qpos[:1]) / self.dt,
            position[1:],
            np.clip(velocity, -10, 10)
        ])
    
    def _get_state(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        self.viewer.cam.lookat[2] = 1.15
        self.viewer.cam.elevation = -20