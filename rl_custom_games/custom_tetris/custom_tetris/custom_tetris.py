import os
import time
from typing import Dict, Any, Optional, Tuple, Union, List, TypeVar
from PIL import Image

import logging
import random
import itertools

import numpy as np
from PIL.Image import Resampling
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
    np.array([[1], [1], [1], [1]])
]

BASIC_BRICKS = [

    np.array([[1, 1]]),

    np.array([[1]]),
]

BASIC_EXT_BRICKS = [

    np.array([[1, 1]]),

    np.array([[1, 1],
              [1, 0]]),

    np.array([[1]]),
]

BASIC_EXT_BRICKS2 = [

    np.array([[1, 1]]),

    np.array([[1, 1],
              [1, 0]]),
    np.array([[0, 1],
              [1, 0]]),

    np.array([[1]]),
]

BRICKS_MAP = {"traditional": BRICKS,
              "basic": BASIC_BRICKS,
              "basic_ext": BASIC_EXT_BRICKS,
              "basic_ext2": BASIC_EXT_BRICKS2,
              }

PRINTMODE = False

logger = logging.getLogger("ttenv")


def rand_birck(randgen,bset):
    bidx = randgen.randint(0,len(bset)-1)
    return bidx,bset[bidx]


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

    spec = EnvSpec("FgTetris-v0")

    # _np_random: Optional[np.random.Generator] = None

    def __init__(self, board_height=14, board_width=7, brick_set="traditional", max_step=100, seed=None,format_as_onechannel=True):
        self.current_brick_idx = None
        self.score = 0
        self.reward_range = (-10, max_step)
        if seed is None:
            self.seed = random.randint(0,9999999)
        else:
            self.seed = seed
        self.rand_generator = random.Random(self.seed)

        self.brick_set = BRICKS_MAP[brick_set]

        self.BOARD_SHAPE = (board_height, board_width)
        self.brick_location = None
        self.current_brick = None
        self.back_board = np.zeros(shape=self.BOARD_SHAPE, dtype=np.int8)
        self.take_brick_on_top()

        self.output_height = board_height
        self.output_width = board_width

        self.format_as_onechannel = format_as_onechannel
        depth = 1 if format_as_onechannel else 3
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(self.output_height, self.output_width, depth),
                                            dtype=np.uint8)

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


        self.score += (removed)**2

        pen = 0
        for col in range(self.output_width):
            blocks = False
            column = self.back_board[:,col]
            for row in column:
                if row == 1:
                    blocks=True
                elif blocks:
                    pen+=1

        self.score -= pen/10

        white_block_above_4 = np.sum(self.back_board[:-2])
        self.score -= white_block_above_4 / ((self.output_height-2)*self.output_width)
#
        #count_by_row = np.sum(self.back_board[:- 2], 1)
        #for vidx,v in enumerate(count_by_row):
        #    if v>0:
        #        break
#
        #pen = (self.output_height - vidx) / self.output_height
        #self.score -= pen*v



    def take_brick_on_top(self):  # take a random brick and add it to the board
        _limit = 6
        white_block_above_limit = np.sum(self.back_board, 1)[:-_limit].sum()
        if white_block_above_limit > 0:
            return self.latest_obs, -1, True, {"new_brick":self.current_brick_idx}


        self.current_brick_idx , self.current_brick = rand_birck(self.rand_generator,self.brick_set)
        self.brick_location = ((self.BOARD_SHAPE[1] // 2) - 1, 0)

        obs = self.back_board.copy()
        set_result = set_at(obs, self.brick_location, self.current_brick * 2)
        if set_result == 1:
            logger.debug("with new block")
            self.latest_obs = obs
            debug_render(self.latest_obs)
            return self.latest_obs, self.score, False, {"new_brick":self.current_brick_idx}
        else:
            logger.debug("stop score: %s", self.score)
            self.latest_obs = obs
            debug_render(self.latest_obs)
            return self.latest_obs, -1, True, {"new_brick":self.current_brick_idx}

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

        return self.return_formater(*self.take_brick_on_top())[0]

    def return_formater(self, observation, score, stop, info):

        if self.format_as_onechannel:

            return observation, score, stop, info
            # return observation.flatten(),score, stop, info
            obs = Image.fromarray((observation * 127).astype(np.uint8), "L")

            obs = obs.resize((self.output_width, self.output_height), resample=Resampling.NEAREST, box=None, reducing_gap=None)
            # obs.save(f"{time.time()}.png")
            obs = np.uint8(obs)

            return (np.expand_dims(obs, 2), score, stop, info)
        else:

            obs=observation.astype(np.uint8)
            n_values = 3
            buff= np.eye(n_values)[obs]
            return (buff, score, stop, info)


    def render(self, mode="print"):
        lines = debug_print(self.latest_obs)
        lines[-1] = "rew: %.2f" % self.score
        if mode=="txt":
            return lines
        else:
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


class GroupedActionSpace(CustomTetris):

    def __init__(self,*args,**kwargs):
        CustomTetris.__init__(self,*args,**kwargs)


        lefts= reversed([[1]*_ for _ in range(1,self.output_width//2 )])

        rights = [[2]*_ for _ in range(1,self.output_width//2 +2)]

        t=[]
        for l in lefts:
            t.append(l+[3])
        t.append([3])
        for l in rights:
            t.append(l+[3] ) 
        
        # 4 rotate
        # 3 push down
        self.allcombo = []
        for r in [[], [4], [4, 4], [4, 4, 4]]:
            for p in t:
                self.allcombo.append(r + p)

        self.action_space = spaces.Discrete(len(self.allcombo))

        self.brick_ohe = np.eye(len(self.brick_set)).astype(np.uint8)
        shape = len(self.brick_ohe[0]) + ((6) * self.output_width)
        self.observation_space = spaces.Box(low=0, high=255,
                                            shape=(shape,),
                                            dtype=np.uint8)
    def reset(self):
        obs = CustomTetris.reset(self)

        return self._remakeobs(self.current_brick_idx,obs)

    def _remakeobs(self,brick_idx,obs):
        brickohe = (self.brick_ohe[brick_idx] * 254)
        obsflat = obs[-6:].flatten()*254
        return np.concatenate([brickohe, obsflat])
    def step(self, action):

        suite = self.allcombo[action]

        cum_rew = 0
        for action_item in suite:
            obs,rew,finished,info = CustomTetris.step(self,action_item)

            if finished:
                return self._remakeobs(0, obs),cum_rew,finished,info
            else:
                if "new_brick" in info:
                    newobs = self._remakeobs(info["new_brick"], obs)
                    return newobs,rew,finished,info
                else:
                    cum_rew+=rew

        raise("errr")
