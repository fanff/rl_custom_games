

rm -rf mlruns/0
seq 32 | xargs -I% -P 16 python rl_custom_games/custom_tetris/bin/hp_A2Ctt.py --device="cuda:" --pidx %