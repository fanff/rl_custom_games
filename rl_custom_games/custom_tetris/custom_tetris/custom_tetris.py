import os
from typing import Dict, Any, Optional, Tuple, Union, List, TypeVar

import logging
import random
import itertools

import numpy as np
from gym import Env
from gym.envs.registration import EnvSpec
from gym.vector.utils import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

BRICKS = [
    np.array([[1, 1],
              [1, 1]]),
    np.array([[0, 1, 1],
              [1, 1, 0]]),
    np.array([[1, 1, 0],
              [0, 1, 1]]),
    np.array([[0, 1, 0],
              [1, 1, 1]]),
    np.array([[1, 0, 0],
              [1, 1, 1]]),
    np.array([[0, 0, 1],
              [1, 1, 1]]),
    np.array([[1, 1, 1, 1]])
]

BASIC_BRICKS = [

    np.array([[1, 1]]),

    np.array([[1]]),
]

PRINTMODE = False

logger = logging.getLogger("ttenv")


def rand_birck(bset):
    return random.choice(bset)


def rotate_brick(inp):
    out = np.zeros(shape=(inp.shape[1], inp.shape[0]), dtype=np.int8)
    for ridx, r in enumerate(inp):
        for cidx, c in enumerate(r):
            out[cidx][inp.shape[0] - ridx - 1] = c
    return out


def set_at_dry(board, blick_location, brick):
    boardh, boardw = board.shape
    w, h = blick_location
    br, bw = brick.shape

    if h + br - 1 >= boardh or h < 0:
        # logger.debug("fix")
        return -1

    if w + bw - 1 >= boardw or w < 0:
        # logger.debug("outofrange")
        return -2

    indexes = (itertools.product(np.arange(h, h + br), np.arange(w, w + bw)))
    for (x, y), v in zip(indexes, brick.reshape(br * bw)):
        if v == 0: continue
        if board[x][y] > 0 and v > 0:
            return -3
    return 1


def set_at(board, blick_location, brick):
    boardh, boardw = board.shape
    w, h = blick_location
    br, bw = brick.shape

    if h + br - 1 >= boardh or h < 0:
        # logger.debug("fix")
        return -1

    if w + bw - 1 >= boardw or w < 0:
        # logger.debug("outofrange")
        return -2

    indexes = (itertools.product(np.arange(h, h + br), np.arange(w, w + bw)))
    for (x, y), v in zip(indexes, brick.reshape(br * bw)):
        if v == 0: continue
        if board[x][y] > 0 and v > 0:
            return -3
        board[x][y] = v

    return 1


def check_rows(board):
    rowstodrop = list((rowidx for rowidx, row in enumerate(board) if np.sum(row) >= board.shape[1]))

    if len(rowstodrop) == 0:
        return board, 0
    else:
        top = np.zeros((len(rowstodrop), board.shape[1]))
        tmp = np.concatenate([board[i] for i in range(len(board)) if i not in rowstodrop]).reshape(
            (len(board) - len(rowstodrop), board.shape[1]))

        return np.concatenate([top, tmp]), len(rowstodrop)


class CustomTetris(Env):
    # Set this in SOME subclasses
    metadata: Dict[str, Any] = {"render.modes": ["print"]}
    # define render_mode if your environment supports rendering
    render_mode: Optional[str] = None
    reward_range = (-float("inf"), float("inf"))
    spec = EnvSpec("FgTetris-v0")

    # Set these in ALL subclasses
    action_space = spaces.Discrete(5)
    observation_space = None
    # Created
    _np_random: Optional[np.random.Generator] = None

    def __init__(self, board_height=14, board_width=7, brick_set="traditional",max_step=100):
        self.score = 0
        if brick_set == "traditional":
            self.brick_set = BRICKS
        else:
            self.brick_set = BASIC_BRICKS

        self.BOARD_SHAPE = (board_height, board_width)
        self.brick_location = None
        self.current_brick = None
        self.back_board = np.zeros(shape=self.BOARD_SHAPE, dtype=np.int8)
        self.take_brick_on_top()

        self.observation_space = spaces.Box(low=0, high=2,
                                            shape=(board_height * board_width,),
                                            dtype=np.int8)

        self.action_space = spaces.Discrete(5)

        self.step_count = 0
        self.max_step = max_step

    def fix_on_back_board(self):
        # self.score += 0.01
        obs = self.back_board.copy()
        set_at(obs, self.brick_location, self.current_brick)
        self.back_board = obs
        #
        self.back_board, removed = check_rows(self.back_board)
        self.score += removed

    def take_brick_on_top(self):  # take a random brick and add it to the board

        self.current_brick = rand_birck(self.brick_set)
        self.brick_location = ((self.BOARD_SHAPE[1] // 2) - 1, 0)

        obs = self.back_board.copy()
        set_result = set_at(obs, self.brick_location, self.current_brick * 2)
        if set_result == 1:
            logger.debug("with new block")
            self.latest_obs = obs
            debug_render(self.latest_obs)
            return self.latest_obs, self.score, False, {}
        else:
            logger.debug("stop score: %s", self.score)
            self.latest_obs = obs
            debug_render(self.latest_obs)
            return self.latest_obs, -1, True, {}

    def return_formater(self, obs, score, stop, info):
        return obs.flatten(), score, stop, info

    def step(self, action):
        if self.step_count > self.max_step:
            r = self.return_formater(self.latest_obs, 0, True, {})
            return r
        else:
            self.step_count += 1
        self.score = 0

        if action == 4:  # rotate
            logger.debug("rotate")
            rotated = rotate_brick(self.current_brick)
            if set_at_dry(self.back_board, self.brick_location, rotated) == 1:
                self.current_brick = rotated

        if action == 1:  # left

            logger.debug("left")
            candidate_loc = (self.brick_location[0] - 1, self.brick_location[1])
            set_result = set_at_dry(self.back_board, candidate_loc, self.current_brick * 2)
            if set_result == 1:
                self.brick_location = candidate_loc

        if action == 2:  # right
            logger.debug("right")
            candidate_loc = (self.brick_location[0] + 1, self.brick_location[1])
            set_result = set_at_dry(self.back_board, candidate_loc, self.current_brick * 2)
            if set_result == 1:
                self.brick_location = candidate_loc

        if action == 3:  # down
            logger.debug("down")

            iterin = True
            while iterin:
                candidate_loc = (self.brick_location[0], self.brick_location[1] + 1)
                if set_at_dry(self.back_board, candidate_loc, self.current_brick) == 1:
                    self.brick_location = candidate_loc
                else:
                    iterin = False
            self.fix_on_back_board()
            return self.return_formater(*self.take_brick_on_top())

        if action != 3:
            # if action was 0,1,2,4 we proceed to a down step
            obs = self.back_board.copy()
            candidate_loc = (self.brick_location[0], self.brick_location[1] + 1)
            set_result = set_at(obs, candidate_loc, self.current_brick * 2)
            if set_result == 1:
                self.brick_location = candidate_loc
                self.latest_obs = obs
                debug_render(self.latest_obs)
                return self.return_formater(self.latest_obs, self.score, False, {})
            else:
                self.fix_on_back_board()
                return self.return_formater(*self.take_brick_on_top())

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):

        logger.debug("reset!")
        self.score = 0
        self.step_count = 0
        self.back_board = np.zeros(shape=self.BOARD_SHAPE, dtype=np.int8)
        self.take_brick_on_top()

        return self.back_board.flatten()

    def render(self, mode="print"):

        lines = debug_print(self.latest_obs)
        lines[-1] = "score: %.2f" % self.score
        print("\n".join(lines))

    def close(self):
        """Override close in your subclass to perform any necessary cleanup.
        Environments will automatically :meth:`close()` themselves when
        garbage collected or when the program exits.
        """
        pass


def debug_print(board):
    mapper = {0: "  ", 1: "▒▒", 2: "██"}
    lines = []
    for row in board:
        mid = "".join(mapper[elem] for elem in row)
        lines.append(f"│{mid}│")
    lines.append("---")
    return lines


def debug_render(board):
    mapper = {0: "  ", 1: "▒▒", 2: "██"}
    for row in board:
        mid = "".join(mapper[elem] for elem in row)
        logging.debug(f"│{mid}│")
    logging.debug("-----")


def find_latest(path="./logs/ttppo/"):
    mdlname = (os.path.join(path, _) for _ in os.listdir(path) if _.endswith(".zip"))

    mdlname = [(os.path.getmtime(i), i) for i in mdlname]
    mdlname = list(sorted(mdlname, key=lambda x: x[0]))

    return mdlname[-1][1]


class VecTetris(DummyVecEnv):
    def render(self, mode="print"):
        for _ in self.envs:
            _.render(mode)
