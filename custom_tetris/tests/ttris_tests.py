import time
import unittest

import numpy as np

from ..custom_tetris.custom_tetris import rotate_brick, set_at, set_at_dry


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
        np.zeros(shape=(10,10))

        buff = np.zeros(shape=(10, 10))
        buffclone = np.zeros(shape=(10, 10))
        res = []
        for i in range(10000):

            strt = time.time()

            buffclone = buff.copy()
            set_at(buffclone,(3,3),a)
            end = time.time()
            res.append(end-strt)


        print(sum(res))



if __name__ == '__main__':
    unittest.main()
