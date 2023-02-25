


seq 32 | xargs -I% -P 8 python rl_custom_games/custom_tetris/bin/hp_A2Ctt.py --device="cuda:" --pidx %