import abc
from typing import Tuple, Optional, Any, Dict

import gym
import numpy as np
from tqdm import tqdm

from cogrip import render
from cogrip.core.engine import Engine
from cogrip.pentomino.config import PentoConfig
from cogrip.pentomino.objects import Piece
from cogrip.pentomino.state import Board

from cogrip.tasks import TaskLoader, Task

EPISODE_STEP_COUNT = "episode/step/count"
EPISODE_OUTCOME_FAILURE = "episode/outcome/failure"
EPISODE_OUTCOME_SUCCESS = "episode/outcome/success"
EPISODE_OUTCOME_ABORT = "episode/outcome/abort"

GRIPPER_ID = 0


class CoGripEnv(gym.Env, abc.ABC):
    metadata = {"render.modes": ["human", "rgb_array"], "video.frames_per_second": 25}

    def __init__(self, task_loader: TaskLoader, hparams: Dict):
        self.task_loader = task_loader
        self.hparams = hparams

        self.engine = Engine(PentoConfig())  # reset the state on each env reset (e.g. size of the maps)
        self.debug = hparams["debug"] if "debug" in hparams else False

        """ Agents """
        assert "speaker.vision.channels" in self.hparams
        assert hparams["speaker_spec"] is not None, "No speaker_spec given"
        speaker_kwargs = hparams["speaker_kwargs"]
        self.speaker = hparams["speaker_spec"].create(speaker_kwargs)

        assert "follower.vision.channels" in self.hparams
        assert "follower.agent.view_size" in self.hparams
        assert hparams["follower_spec"] is not None, "No follower_spec given"
        follower_kwargs = hparams["follower_kwargs"]
        self.follower = hparams["follower_spec"].create(follower_kwargs)

        """ State """
        self.current_task: Task = None
        self.current_board: Board = None

        self.step_count = 0
        self.info = {}

        self.progress_bar = None
        if "env.progress.show" in self.hparams and self.hparams["env.progress.show"]:
            progress_length = len(task_loader)
            if "env.progress.length" in self.hparams:
                progress_length = self.hparams["env.progress.length"]
            self.progress_bar = tqdm(total=progress_length)

    def log_debug(self, message):
        if self.debug:
            print(message)

    def seed(self, seed=None):
        pass

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Any:
        self.log_debug("\n NEW EPISODE")
        self.info = {}
        if self.progress_bar is not None:
            self.progress_bar.update(1)
        """ We go through the tasks; and restart if reached the end """
        self.current_task = self.task_loader.next_task()
        # reset board
        self.current_board = self.current_task.create_board()
        self.engine.set_state(self.current_board)  # this new board has no gripper yet
        self.engine.add_gr(GRIPPER_ID)  # we expect exactly one gripper (that can be controlled by an agent)
        # reset speaker
        self.follower.reset(self.current_task, self.current_board, self.current_gripper_coords, self.info)
        self.speaker.reset(self.current_task, self.current_board, self.current_gripper_coords, self.info)
        self.step_count = 0
        obs = self.on_reset()
        self.info.clear()  # remove infos (if accidentally added during reset)
        if self.current_task.idx is not None:
            self.info["task/idx"] = self.current_task.idx
        return obs

    @abc.abstractmethod
    def on_reset(self) -> Any:
        pass

    def render(self, mode='rgb_array', channel_first=False):
        if mode == 'rgb_array':
            return render.to_rgb_array(self.current_board,
                                       self.current_task.target_piece,
                                       self._current_gripper_coords(),
                                       num_channels=3,
                                       channel_first=channel_first)  # return RGB frame suitable for video
        elif mode == 'human':
            return render.to_rgb_array(self.current_board,
                                       self.current_task.target_piece,
                                       self._current_gripper_coords(),
                                       num_channels=3, channel_first=channel_first)
        else:
            super().render(mode=mode)  # just raise an exception

    def step(self, speaker_action: object) -> Tuple[Any, float, bool, dict]:
        # speaker_action = 0  # set to always silence
        # speaker_action = np.random.choice([8, 9, 10, 11, 12, 13])  # always choose a reference
        self.step_count += 1

        obs, piece_gripped = self.on_step(speaker_action)

        if self.is_abort_condition():
            self.info[EPISODE_OUTCOME_ABORT] = 1
            self.info[EPISODE_OUTCOME_SUCCESS] = 0
            self.info[EPISODE_OUTCOME_FAILURE] = 0
            self.info[EPISODE_STEP_COUNT] = self.step_count
            done = True
            reward = self.on_failure_reward()
        elif self.is_terminal_condition(piece_gripped):
            self.info[EPISODE_OUTCOME_ABORT] = 0
            self.info[EPISODE_STEP_COUNT] = self.step_count
            done = True
            if self._is_target_piece(piece_gripped):
                self.info[EPISODE_OUTCOME_SUCCESS] = 1
                self.info[EPISODE_OUTCOME_FAILURE] = 0
                reward = self.on_success_reward()
            else:
                self.info[EPISODE_OUTCOME_SUCCESS] = 0
                self.info[EPISODE_OUTCOME_FAILURE] = 1
                reward = self.on_failure_reward()
        else:
            done = False
            reward = self.on_step_reward()

        # future package gymnasium will have a distinction between terminated and truncated
        self.info["done"] = done
        return obs, reward, done, self.info

    @abc.abstractmethod
    def on_step(self, speaker_action: object) -> Tuple[Dict, Piece]:
        pass

    @abc.abstractmethod
    def on_success_reward(self) -> float:
        pass

    @abc.abstractmethod
    def on_failure_reward(self) -> float:
        pass

    @abc.abstractmethod
    def on_step_reward(self) -> float:
        pass

    def _gen_follower_vision(self, return_pixels=True):  # for training use channel_first
        overview = render.to_rgb_array(self.current_board,
                                       self.current_task.target_piece,
                                       self._current_gripper_coords(),
                                       num_channels=self.hparams["follower.vision.channels"],
                                       return_pixels=return_pixels,
                                       channel_first=True)
        fov = render.compute_fov(overview, self.current_board, self._current_gripper_coords(),
                                 fov_size=self.hparams["follower.agent.view_size"],
                                 channel_first=True)
        return fov

    def _gen_speaker_vision(self):  # for training use channel_first
        overview = render.to_rgb_array(self.current_board,
                                       self.current_task.target_piece,
                                       self._current_gripper_coords(),
                                       num_channels=self.hparams["speaker.vision.channels"],
                                       channel_first=True,
                                       padding=self.hparams["speaker.agent.max_size"] - self.map_size)
        return overview

    def _gen_gripper_coords(self):
        """ Scale coords from (0,N) into (-1,+1) with (0,0) as the center"""
        x, y = self._current_gripper_coords()
        x = x / self.current_task.grid_config.width
        y = y / self.current_task.grid_config.height
        x = np.around((x - 0.5) * 2, 5)
        y = np.around((y - 0.5) * 2, 5)
        return np.array([x, y])

    @property
    def current_gripper_coords(self) -> Tuple[int, int]:
        return self._current_gripper_coords()

    @property
    def map_size(self) -> int:
        return self.current_board.grid_config.width

    def _current_gripper_coords(self) -> Tuple[int, int]:
        """ Returns (x,y) """
        x, y = self.engine.get_gripper_coords(GRIPPER_ID)
        return int(x), int(y)

    def is_terminal_condition(self, piece_gripped: Piece):
        return piece_gripped is not None

    def _is_target_piece(self, piece_gripped: Piece):
        if piece_gripped is None:
            return False
        return self.current_task.target_piece.id_n == piece_gripped.id_n

    def is_abort_condition(self):
        return self.step_count >= self.get_max_steps()

    def get_max_steps(self):
        return self.current_task.max_steps

    def _grip(self) -> Piece:
        self.engine.grip(GRIPPER_ID)
        piece = self.engine.get_gripped_obj(GRIPPER_ID)
        return piece

    def _move(self, dx: int, dy: int):
        return self.engine.mover.apply_movement(self.engine, "move", GRIPPER_ID, x_steps=dx, y_steps=dy)

    def close(self):
        pass
