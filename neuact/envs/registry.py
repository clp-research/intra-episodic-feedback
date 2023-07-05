from typing import Type, Dict

from cogrip.registry import register_with_hparams
from cogrip.tasks import TaskLoader
from neuact.agents.speakers import SpeakerSpec
from neuact.hparams import default_params


def register_pento(task_loader: TaskLoader, return_feedback: bool = True,
                   speaker_spec: Type[SpeakerSpec] = None, speaker_kwargs: Dict = None):
    env_name = f"PentoEnv"
    if "preference_order" in speaker_kwargs:
        env_name += f"-{speaker_kwargs['preference_order']}"
    if not return_feedback:
        env_name += "-nofb"
    else:
        env_name += "-fb"
    hparams = default_params(return_feedback, speaker_spec, speaker_kwargs)
    return register_with_hparams(task_loader, hparams, env_name)
