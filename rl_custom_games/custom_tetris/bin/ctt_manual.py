import click

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris, GroupedActionSpace


@click.command()
@click.option("--board_height", default=12, type=int,show_default=True)
@click.option("--board_width", default=6, type=int,show_default=True)
@click.option("--brick_set", default="basic", type=str,show_default=True)
@click.option("--max_step", default=2000, type=int,show_default=True)
def play_tetris(board_height,board_width,brick_set,max_step):
    evalenv = GroupedActionSpace(board_height,board_width,brick_set,max_step=max_step,seed=4)

    evalenv.reset()
    evalenv.render("df")
    for i in range(100000):
        print(evalenv.action_space)
        r = input("action ?")


        obs, reward, done, info = evalenv.step(int(r))
        evalenv.render("df")
        # VecEnv resets automatically
        if done:
            obs = evalenv.reset()

if __name__ == "__main__":
    play_tetris()