from typing import Type, Dict

from neuact.agents import NoopFollower
from neuact.agents.speakers import SpeakerSpec


def default_params(return_feedback: bool = True, speaker_spec: Type[SpeakerSpec] = None, speaker_kwargs: Dict = None):
    return {
        "return_feedback": return_feedback,
        "speaker_spec": speaker_spec,
        "speaker_kwargs": speaker_kwargs,
        "speaker.vision.channels": None,
        "follower_spec": NoopFollower,  # this will be the trainable agent
        "follower_kwargs": dict(view_size=11),
        "follower.vision.channels": 3,
        "follower.agent.view_size": 11,
        "env.entry.point": "neuact.envs.follower_env:PentoFollowerEnv"
    }
