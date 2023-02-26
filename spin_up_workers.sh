

rm -rf mlruns/0
seq 32 | xargs -I% -P 8 python rl_custom_games/custom_tetris/bin/gen_hp.py --device="cuda:" --pidx %