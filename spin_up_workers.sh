


seq 16 | xargs -I% -P 6 python rl_custom_games/custom_tetris/bin/hp_tt.py --device="cuda:" --pidx %