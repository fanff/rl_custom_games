import random
import time
import unittest

import pandas as pd


class MyTestCase(unittest.TestCase):
    def test_speed(self):
        from ..custom_tetris.custom_tetris import CustomTetris

        env = CustomTetris(30,20)
        res = []
        env.reset()
        for i in range(1000):
            action = random.randint(0,env.action_space.n-1)
            strt= time.time()
            obs,r,stop,info = env.step(action)
            end = time.time()

            res.append([end,strt])
            if stop:
                env.reset()

        df = pd.DataFrame(res,columns=["end","str"])
        df["dur"] = df["end"] - df["str"]
        print((df[["dur"]]*1000).describe())



if __name__ == '__main__':
    unittest.main()
