from stable_baselines3.common.env_checker import check_env

from rl_custom_games.custom_tetris.custom_tetris.custom_tetris import CustomTetris

env = CustomTetris()
check_env(env, warn=True, skip_render_check=False)
