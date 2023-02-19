import unittest

from rl_custom_games.schedules import pow_schedule


class MyTestCase(unittest.TestCase):
    def test_something(self):
        l = pow_schedule(2.0, 1.0)
        self.assertAlmostEqual(l(1.0), 2.0)
        self.assertAlmostEqual(l(0.0),1.0)


if __name__ == '__main__':
    unittest.main()
