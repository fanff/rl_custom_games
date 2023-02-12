import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
# from gym_derk.envs import DerkEnv

#env = DerkEnv(
#  n_arenas=1 # Change to 100 for instance, to run many instances in parallel
#)
from stable_baselines3.common.atari_wrappers import AtariWrapper

[print(_) for _ in list(gym.envs.registry.all()) if "Breakout" in str(_)]

env_name = "ALE/Breakout-v5)"

print("loading ",env_name)

env = make_vec_env(env_name, n_envs=8)

env = AtariWrapper(env,frame_skip= 4)
print("wrapped")
#env = gym.make("CartPole-v1")
model = A2C("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=500_000)

vec_env = model.get_env()
obs = vec_env.reset()
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    vec_env.render()
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()