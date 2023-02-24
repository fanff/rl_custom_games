import logging
import time
import unittest

import gym
import numpy as np
from gym.utils.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

from ..custom_tetris.custom_tetris import rotate_brick, set_at, set_at_dry, VecTetris


class Test_rotation(unittest.TestCase):
    def test_rotate(self):

        a = np.array([[1, 1, 0],
                  [0, 1, 1]])
        b = rotate_brick(a)
        c = rotate_brick(b)
        print(a)
        print(b)
        print(c)

class Test_set_at(unittest.TestCase):
    def test_set_at(self):

        a = np.array([[1, 1, 0],
                  [0, 1, 1]])

        buffclone = np.zeros(shape=(10, 10))
        set_at(buffclone,(3,3),a)
        set_at_dry(buffclone, (3, 3), a)

class Test_set_at(unittest.TestCase):
    def test_check_env(self):

        from ..custom_tetris.custom_tetris import CustomTetris

        env = CustomTetris()
        check_env(env, warn=True, skip_render_check=False)

    def test_vec_venv(self):
        from ..custom_tetris.custom_tetris import CustomTetris
        #env = make_vec_env(CustomTetris, n_envs=8)
        env = SubprocVecEnv([CustomTetris]*4)
        env.reset()

        env.step([0]*4)
        env.step([3] * 4)
    def test_custom_vec(self):

        from ..custom_tetris.custom_tetris import CustomTetris
        count = 8

        env = VecTetris([CustomTetris]*count)
        env.reset()
        env.render()

        env.step( [0] *count)
        env.render()
        env.step( [3] * count)
        env.render()

    def test_formating(self):

        from ..custom_tetris.custom_tetris import CustomTetris
        ct = CustomTetris(format_as_onechannel=False)

        a = ct.observation_space.sample()

        formated = ct.return_formater(ct.latest_obs,None,None,None)


        print(formated[0].shape)

        ct.reset()

        ct.step(3)
        ct.step(3)
        ct.step(3)

if __name__ == '__main__':
    unittest.main()
