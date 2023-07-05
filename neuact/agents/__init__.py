import abc
from typing import Tuple, Dict

from cogrip.pentomino.state import Board
from cogrip.tasks import Task
from neuact.envs import Actions


class Follower(abc.ABC):

    def __init__(self, view_size: int, **kwargs):
        self.view_size = view_size
        self.task = None
        self.board = None
        self.current_pos = None
        self.info = None
        self.debug = kwargs["debug"] if "debug" in kwargs else False

    def log_debug(self, message):
        if self.debug:
            print(f"{self.__class__.__name__}: {message}")

    def reset(self, task: Task, board: Board, gripper_pos: Tuple[int, int], info: Dict = None):
        self.task = task
        self.board = board
        self.current_pos = gripper_pos
        self.info = info

    def log_info(self, key, value):
        if self.info:
            self.info[key] = value

    @abc.abstractmethod
    def forward(self, obs, gripper_pos: Tuple[int, int]) -> Actions:
        pass


class FollowerSpec:
    """ Use this for the gym env registry to later create a specific instance """

    @classmethod
    def get_name(cls):
        return cls.__name__

    @classmethod
    def create(cls, kwargs=None) -> Follower:
        raise NotImplementedError()


class NoopFollower(Follower, FollowerSpec):

    def forward(self, obs, gripper_pos: Tuple[int, int]) -> Actions:
        pass

    @classmethod
    def create(cls, kwargs=None) -> Follower:
        return NoopFollower(**kwargs)
