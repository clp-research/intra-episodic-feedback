from neuact.agents.speakers import Speaker, SpeakerSpec, SILENCE
from neuact.agents.speakers.feedback import PieceFeedbackSpeaker, ThresholdedPieceFeedbackSpeaker
from neuact.agents.speakers.mission import IAMissionSpeaker, MissionSpeaker


class PieceFeedbackIAMissionSpeaker(PieceFeedbackSpeaker, IAMissionSpeaker, SpeakerSpec):

    def __init__(self, **kwargs):
        super(PieceFeedbackIAMissionSpeaker, self).__init__(**kwargs)

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "PieceFeedbackIAMissionSpeaker requires kwargs"
        return cls(**kwargs)


class PositivePieceFeedbackIAMissionSpeaker(PieceFeedbackSpeaker, IAMissionSpeaker, SpeakerSpec):
    """ Only utters 'yes this piece' in addition to the mission statement """

    def __init__(self, **kwargs):
        super(PositivePieceFeedbackIAMissionSpeaker, self).__init__(**kwargs)

    def _on_gripper_over_other(self) -> (str, int):
        return SILENCE, 0

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "PositivePieceFeedbackIAMissionSpeaker requires kwargs"
        return cls(**kwargs)


class ThresholdedPieceFeedbackIAMissionSpeaker(ThresholdedPieceFeedbackSpeaker, IAMissionSpeaker, SpeakerSpec):
    def __init__(self, distance_threshold: int, time_threshold: int, preference_order: str):
        super(ThresholdedPieceFeedbackIAMissionSpeaker, self).__init__(distance_threshold=distance_threshold,
                                                                       time_threshold=time_threshold,
                                                                       preference_order=preference_order)

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "ThresholdedPieceFeedbackIAMissionSpeaker requires kwargs"
        return cls(**kwargs)


class StepwisePieceFeedbackMissionSpeaker(ThresholdedPieceFeedbackSpeaker, MissionSpeaker, SpeakerSpec):
    def __init__(self, distance_threshold: int, time_threshold: int):
        super().__init__(distance_threshold, time_threshold)

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        return cls(distance_threshold=0, time_threshold=0)


class StepwisePieceFeedbackIAMissionSpeaker(ThresholdedPieceFeedbackSpeaker, IAMissionSpeaker, SpeakerSpec):
    def __init__(self, distance_threshold: int, time_threshold: int):
        super().__init__(distance_threshold, time_threshold)

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        return cls(distance_threshold=0, time_threshold=0)
