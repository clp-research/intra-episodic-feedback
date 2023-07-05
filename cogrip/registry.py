from typing import Dict

import gym
from stable_baselines3.common.env_util import make_vec_env

from cogrip.tasks import TaskLoader
from cogrip.wrappers import PartialVisionObsWrapper

ENV_VERSION = "v1"


def check_env_name(env_name, env_version: str = None):
    version = ENV_VERSION
    if env_version is not None:
        version = "v" + env_version
    if not env_name.endswith(f"-{version}"):
        env_name = env_name + f"-{version}"
    return env_name


def make_env(env_name, use_limited_fov=False):
    env_name = check_env_name(env_name)
    env = gym.make(env_name)
    if use_limited_fov:
        env = PartialVisionObsWrapper(env)
    return env


def make_envs(env_name, num_envs, use_limited_fov=False):
    def _wrap(env):
        if use_limited_fov:
            print("Use limited field-of-view wrapper...")
            env = PartialVisionObsWrapper(env)
        return env

    if env_name is None:
        return None
    env_name = check_env_name(env_name)
    envs = make_vec_env(env_name, n_envs=num_envs, wrapper_class=_wrap)
    return envs


def make_envs_with_hparams(env_name: str, hparams: Dict):
    return make_vec_env(env_name, n_envs=hparams["num_envs"], wrapper_class=None)


def register_with_hparams(task_loader: TaskLoader, hparams: Dict, env_name=f"CoGripEnv"):
    env_name = check_env_name(env_name)
    gym.register(env_name,
                 kwargs={"task_loader": task_loader, "hparams": hparams},
                 entry_point=hparams['env.entry.point'])
    return env_name
