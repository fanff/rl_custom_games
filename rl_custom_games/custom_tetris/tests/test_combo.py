import copy
import unittest

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import GroupedActionSpace
from rl_custom_games.schedules import pow_schedule


class MyTestCase(unittest.TestCase):
    def test_something(self):
        env= GroupedActionSpace(15, 5, "traditional",
                           500,
                            seed=5,
                           format_as_onechannel=True)

        obs = env.reset()
        assert len(obs) == 5*6 + 7


        obs, rew,finished,dict = env.step(0)
        assert len(obs) == 5 * 6 + 7

        env = GroupedActionSpace(15, 4, "traditional",
                                 500,
                                 seed=5,
                                 format_as_onechannel=True)

        candidate_action = range(env.action_space.n)
        env_clone = (copy.deepcopy(env) for _ in range(env.action_space.n))

        results = []
        for candidate_action,cloned_env in zip(candidate_action,env_clone):
            obs, rew, finished, dict = cloned_env.step(candidate_action)
            results.append( (candidate_action, cloned_env.render("txt")) )

        txt_line_count = len(results[0][1])
        line_size = len(results[0][1][0])
        for i in range(-1,txt_line_count):
            if i == -1:
                actionsline = "".join( str(act).ljust(line_size+1) for act, obs in results)
                print(actionsline)

            else:
                game_line = "".join( obs[i].ljust(line_size+1) for act, obs in results)
                print(game_line)


    def test_deepcrack(self):

        action_list=[]
        env = GroupedActionSpace(15, 7, "traditional",
                                 500,
                                 seed=1,
                                 format_as_onechannel=True)

        obs = env.reset()

        env.render("print")
        for i in range(100):

            candidate_action = range(env.action_space.n)
            env_clone = (copy.deepcopy(env) for _ in range(env.action_space.n))

            results = []
            for candidate_action,cloned_env in zip(candidate_action,env_clone):
                obs, rew, finished, dict = cloned_env.step(candidate_action)
                if not finished:
                    results.append((candidate_action, rew))

            if len(results) == 0:
                raise Exception("not possible")
            best_action = max(results,key=lambda x:x[1])[0]
            env.step(best_action)
            env.render("print")
            action_list.append(best_action)




if __name__ == '__main__':
    unittest.main()