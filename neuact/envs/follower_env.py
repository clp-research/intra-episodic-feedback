from cogrip.env import CoGripEnv
from typing import Tuple, Any, Dict, List, Type, Union

import numpy as np
from gym import spaces

from cogrip.pentomino.objects import Piece
from cogrip.registry import check_env_name
from cogrip.tasks import TaskLoader
from cogrip.wrappers import PartialVisionObsWrapper, FullyPartialObsWrapper
from neuact.agents.speakers.combined import SpeakerSpec
from neuact.envs import Actions
from neuact.hparams import default_params


class PentoFollowerEnv(CoGripEnv):

    @staticmethod
    def create(task_loader: TaskLoader, return_feedback: bool = True,
               speaker_spec: Type[SpeakerSpec] = None, speaker_kwargs: Dict = None):
        hparams = default_params(return_feedback, speaker_spec, speaker_kwargs)
        env = PentoFollowerEnv(task_loader, hparams)
        return PentoFollowerEnv.wrap_partial(env)

    @staticmethod
    def wrap_partial(env):
        return PartialVisionObsWrapper(env)

    @staticmethod
    def wrap_fully_partial(env):
        return FullyPartialObsWrapper(PartialVisionObsWrapper(env))

    def __init__(self, task_loader: TaskLoader, hparams):
        super().__init__(task_loader, hparams)
        env_name = f"PentoEnv"
        speaker_kwargs = hparams["speaker_kwargs"]
        if "preference_order" in speaker_kwargs:
            env_name += f"-{speaker_kwargs['preference_order']}"
        return_feedback = hparams["return_feedback"]
        if not return_feedback:
            env_name += "-nofb"
        else:
            env_name += "-fb"
        env_name = check_env_name(env_name)
        self.name = env_name
        self.task_loader = task_loader
        self.return_feedback = return_feedback

        """ Action space """
        self.action_space: spaces.Space = spaces.Discrete(len(Actions))

        """ Observation space """
        self.channels = self.hparams["follower.vision.channels"]  # 3
        self.agent_view_size = self.hparams["follower.agent.view_size"]  # 11

        dict_spaces = dict()
        dict_spaces["vision"] = spaces.Box(low=0, high=255,
                                           shape=(self.hparams["follower.vision.channels"],
                                                  self.hparams["follower.agent.view_size"],
                                                  self.hparams["follower.agent.view_size"]),
                                           dtype=np.uint8)
        dict_spaces["gr_coords"] = spaces.Box(low=-1, high=1, shape=(2,))

        dict_spaces["mission"] = self.speaker.get_obs_space()
        if return_feedback:
            dict_spaces["feedback"] = self.speaker.get_obs_space()
        self.observation_space = spaces.Dict(dict_spaces)

        self.gripper_pos_history: List[Tuple[int, int]] = None

    def on_reset(self) -> Any:
        self.gripper_pos_history = [self.current_gripper_coords]
        obs = self.gen_obs()
        return obs

    def to_rgb_array(self, channel_first=True):
        rgb_array = self.current_board.to_rgb_array()
        for idx, color in enumerate([200, 150, 100]):  # paint first the older ones, so newer paint over them
            history_pos = 2 - idx
            if len(self.gripper_pos_history) > history_pos:  # first steps are not contained
                x, y = self.gripper_pos_history[history_pos]  # latest is first
                rgb_array[y, x] = color
        # x, y = self.target_piece.x, self.target_piece.y
        # rgb_array[y, x] = [224, 224, 224]
        # x, y = self.target_piece.get_centroid()
        # rgb_array[y, x] = [127, 127, 127]
        if channel_first:
            rgb_array = np.transpose(rgb_array, (2, 0, 1))
        rgb_array = rgb_array.astype(np.uint8)
        return rgb_array

    def render(self, mode='rgb_array', channel_first=False):
        if mode == 'rgb_array':
            return self.to_rgb_array(channel_first=channel_first)  # return RGB frame suitable for video
        elif mode == 'human':
            return self.to_rgb_array(channel_first=channel_first)
        else:
            super().render(mode=mode)  # just raise an exception

    def _gen_mission(self):
        mission = self.speaker.generate_mission()
        return mission

    def _gen_feedback(self):
        feedback, signal = self.speaker.generate_feedback(self.current_gripper_coords)
        if signal != 0:
            self.info["step/feedback"] = 1
        return feedback

    def gen_obs(self):
        # for training use channel_first
        grid = self.render(mode='rgb_array', channel_first=True)
        obs = {"vision": grid}
        mission = self._gen_mission()
        obs["mission"] = mission
        if self.return_feedback:
            feedback = self._gen_feedback()
            obs["feedback"] = feedback
        obs["gr_coords"] = self._gen_gripper_coords()
        return obs

    def on_step(self, action: Union[List, int]) -> Tuple[Dict, Piece]:
        piece_gripped = None
        if action == Actions.wait:
            self.info["step/action/wait"] = 1
            pass
        elif action == Actions.left:
            self.info["step/action/move"] = 1
            self._move(-1, 0)
        elif action == Actions.right:
            self.info["step/action/move"] = 1
            self._move(1, 0)
        elif action == Actions.up:
            self.info["step/action/move"] = 1
            self._move(0, -1)
        elif action == Actions.down:
            self.info["step/action/move"] = 1
            self._move(0, 1)
        elif action == Actions.grip:
            self.info["step/action/grip"] = 1
            piece_gripped = self._grip()
        else:
            raise ValueError(f"Unknown action: {action}")

        # on reset the initial gripper pos is set,
        # so we set the next on into history here
        # the obs that use this history are only
        # generated later, see gen_obs below
        self.gripper_pos_history.insert(0, self.current_gripper_coords)  # latest  first

        obs = self.gen_obs()
        return obs, piece_gripped

    def _move_reward(self):
        """ Between 0.1 (using all steps) and 0.99 (using only one step) """
        return 1 - 0.9 * (self.step_count / self.current_task.max_steps)

    def on_success_reward(self) -> float:
        return self._move_reward() + 1.

    def on_failure_reward(self) -> float:
        return self._move_reward() - 1.

    def on_step_reward(self) -> float:
        return 0.0
