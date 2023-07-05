import gym
import numpy as np
import torch
from gym import Wrapper, spaces


class StackedVisionObsWrapper(gym.ObservationWrapper):

    def __init__(self, env, num_stack: int, mode: str):
        super().__init__(env)
        print("StackedVisionObsWrapper with mode:", mode, f"({num_stack} frames)")
        assert mode in ["stack_channels", "stack_new_dim"]
        self.num_stack = num_stack
        self.mode = mode
        self.history = []
        unwrapped = self.unwrapped
        vision_space = unwrapped.observation_space["vision"]
        sample = vision_space.sample()
        if mode == "stack_channels":
            vision_shape = (unwrapped.channels * num_stack, sample.shape[1], sample.shape[1])
            vision_space = spaces.Box(low=0, high=255, shape=vision_shape, dtype=np.uint8)
        else:
            vision_shape = (unwrapped.channels, num_stack, sample.shape[1], sample.shape[1])
            vision_space = spaces.Box(low=0, high=255, shape=vision_shape, dtype=np.uint8)
        self.empty = torch.zeros(sample.shape)
        unwrapped.observation_space["vision"] = vision_space

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.history = [self.empty for _ in range(self.num_stack)]
        return self.observation(observation)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.history.insert(0, torch.tensor(observation["vision"]))
        return self.observation(observation), reward, done, info

    def observation(self, observation):
        observation["vision"] = self._stack()
        return observation

    def _stack(self):
        if self.mode == "stack_channels":
            vision_obs = torch.cat(self.history[:self.num_stack], dim=0)
        elif self.mode == "stack_new_dim":
            vision_obs = torch.stack(self.history[:self.num_stack], dim=1)
        else:
            raise ValueError(self.mode)
        return vision_obs

    def render(self, mode="human", **kwargs):
        vision_obs = self.unwrapped.render(mode, **kwargs)
        return vision_obs


class PartialVisionObsWrapper(gym.ObservationWrapper):

    def __init__(self, env, agent_view_size=11):
        super().__init__(env)
        self.agent_view_size = agent_view_size
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        unwrapped = self.unwrapped
        unwrapped.observation_space["vision"] = spaces.Box(low=0, high=255,
                                                           shape=(unwrapped.channels, agent_view_size, agent_view_size),
                                                           dtype=np.uint8)

    def observation(self, observation):
        vision_obs = observation["vision"]
        if vision_obs.shape[1] == self.agent_view_size:  # already reduced e.g. in render
            return observation
        observation["vision"] = self._shrink(vision_obs)
        return observation

    def render(self, mode="human", **kwargs):
        vision_obs = self.unwrapped.render(mode, **kwargs)
        return self._shrink(vision_obs)

    def _shrink(self, vision_obs):
        channel_first = False
        obs_shape = vision_obs.shape
        if obs_shape[0] == 3:
            channel_first = True
        env = self.unwrapped
        max_size = env.current_board.grid_config.width
        x, y = env.current_gripper_coords
        context_size = int((self.agent_view_size - 1) / 2)
        topx = x - context_size
        topy = y - context_size
        if channel_first:
            view = np.full((3, self.agent_view_size, self.agent_view_size),
                           fill_value=0)  # fill with black or "out-of-world"
        else:
            view = np.full((self.agent_view_size, self.agent_view_size, 3),
                           fill_value=0)  # fill with black or "out-of-world"
        for offy in range(self.agent_view_size):
            for offx in range(self.agent_view_size):
                vx = topx + offx
                vy = topy + offy
                if (vx >= 0 and vy >= 0) and (vx < max_size and vy < max_size):
                    if channel_first:
                        arr = vision_obs[:, vy, vx]
                        view[:, offy, offx] = arr
                    else:
                        arr = vision_obs[vy, vx]
                        view[offy, offx, :] = arr
        return view


class FullyPartialObsWrapper(Wrapper):

    def render(self, mode="human", **kwargs):
        if not isinstance(self.env, PartialVisionObsWrapper):
            return self.env.render(mode, **kwargs)
        env = self.unwrapped
        vision_obs = env.render(mode, **kwargs)

        channel_first = self._is_channel_first(vision_obs)
        assert channel_first is False, "FullyPartialObsWrapper should only be used for human rendering"

        max_size = env.current_board.grid_config.width
        x, y = env.current_gripper_coords
        context_size = int((self.agent_view_size - 1) / 2)
        topx = x - context_size
        topy = y - context_size
        for offy in range(self.agent_view_size):
            for offx in range(self.agent_view_size):
                vx = topx + offx
                vy = topy + offy
                if (vx >= 0 and vy >= 0) and (vx < max_size and vy < max_size):
                    coord = vision_obs[vy, vx]
                    if np.all(coord == 255):  # only affect the white pixels
                        vision_obs[vy, vx] = (235, 235, 235)
        return vision_obs

    def _is_channel_first(self, vision_obs):
        channel_first = False
        obs_shape = vision_obs.shape
        if obs_shape[0] == 3:
            channel_first = True
        return channel_first
