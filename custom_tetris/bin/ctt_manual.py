from custom_tetris.custom_tetris.custom_tetris import CustomTetris

evalenv = CustomTetris()

obs = evalenv.reset()
evalenv.render("df")
for i in range(100000):
    r = input("action ?")


    obs, reward, done, info = evalenv.step(int(r))
    evalenv.render("df")
    # VecEnv resets automatically
    if done:
        obs = evalenv.reset()