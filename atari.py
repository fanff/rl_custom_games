
import gym

from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env


from stable_baselines3.common.atari_wrappers import AtariWrapper

[print(_) for _ in list(gym.envs.registry.all()) ]

env_name = "ALE/Tetris-ram-v5"

print("loading ",env_name)

#env = make_vec_env(env_name, n_envs=8)
env = gym.make(env_name)

env = AtariWrapper(env,)
print("wrapped")
#env = gym.make("CartPole-v1")
model = A2C("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=500_000)

vec_env = AtariWrapper(gym.make(env_name,render_mode='human'),)
model.set_env(vec_env)

obs = vec_env.reset()
for i in range(100000):
    print(i)
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)
    #vec_env.render()
    # VecEnv resets automatically
    if done:
      obs = vec_env.reset()