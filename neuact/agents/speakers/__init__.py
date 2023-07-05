from typing import Tuple, Dict

import numpy as np
from gym import spaces

from cogrip.language import encode_sent
from cogrip.pentomino.state import Board
from cogrip.tasks import Task

SILENCE = ""


class Speaker:

    def __init__(self, **kwargs):
        """ Attributes reset at each episode"""
        self.mission: str = None
        self.last_gripper_pos: Tuple[int, int] = None
        self.task: Task = None
        self.board: Board = None
        self.info = None
        """ All speakers share the same vocabulary, but might use only a subset """
        # [color] [shape] [position] -- 12+12+5=29 words
        # ["Take", "the", "piece", "at", "Get", "Select"] -- 6 words
        # <s> <e> <pad> <unk> -- 4 words
        # [Not, this, direction, Yes, there, No, Yeah, way] -- 8 words
        self.vocab_size = 47  # 29+16
        # <s> Take the navy blue F at the top left <e> -- 9 + <s> + <e> = 11
        self.max_sentence_length = 11
        self.obs_space = spaces.Box(low=0, high=self.vocab_size, shape=(self.max_sentence_length,), dtype=np.uint8)

    def _tokenize(self, text: str):
        if text == SILENCE:
            return np.zeros(shape=(self.max_sentence_length,), dtype=np.uint8)  # only padding
        tokens = encode_sent(text, pad_length=self.max_sentence_length)
        return np.array(tokens, dtype=np.uint8)

    def reset(self, task: Task, board: Board, gripper_pos: Tuple[int, int], info: Dict = None):
        self.mission = None
        self.last_gripper_pos = gripper_pos
        self.task = task
        self.board = board
        self.info = info

    def generate_mission(self) -> np.array:
        raise NotImplementedError()

    def generate_feedback(self, gripper_pos: Tuple[int, int], return_string=False) -> (np.array, int):
        raise NotImplementedError()

    def get_obs_space(self) -> spaces.Space:
        return self.obs_space


class SpeakerSpec:
    """ Use this for the gym env registry to later create a specific speaker instance """

    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        raise NotImplementedError()
