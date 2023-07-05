from typing import Tuple

import numpy as np

from cogrip.pentomino.symbolic.algos import PentoIncrementalAlgorithm
from cogrip.pentomino.symbolic.types import PropertyNames
from neuact.agents.speakers import Speaker, SpeakerSpec, SILENCE


class IAMissionSpeaker(Speaker, SpeakerSpec):
    """ A speaker that mentions the shape and color of a piece (but provides no feedback) and ignores positions """

    def __init__(self, preference_order, **kwargs):
        if preference_order == "CSP":
            po = [PropertyNames.COLOR, PropertyNames.SHAPE, PropertyNames.REL_POSITION]
        if preference_order == "CPS":
            po = [PropertyNames.COLOR, PropertyNames.REL_POSITION, PropertyNames.SHAPE]
        if preference_order == "PSC":
            po = [PropertyNames.REL_POSITION, PropertyNames.SHAPE, PropertyNames.COLOR]
        if preference_order == "PCS":
            po = [PropertyNames.REL_POSITION, PropertyNames.COLOR, PropertyNames.SHAPE]
        if preference_order == "SPC":
            po = [PropertyNames.SHAPE, PropertyNames.REL_POSITION, PropertyNames.COLOR]
        if preference_order == "SCP":
            po = [PropertyNames.SHAPE, PropertyNames.COLOR, PropertyNames.REL_POSITION]
        assert po is not None, "preference order should be given, but is None"
        self.reg = PentoIncrementalAlgorithm(po, start_tokens=["take"])  # , "select", "get"])
        print(f"Using IAMissionSpeaker: {preference_order}...")
        super(IAMissionSpeaker, self).__init__(**kwargs)

    def generate_feedback(self, gripper_pos: Tuple[int, int], return_string=False) -> (np.array, int):
        raise NotImplementedError()

    def generate_mission(self):
        if self.mission is None:
            reg = self.reg.generate(self.task.piece_symbols, self.task.target_piece_symbol, is_selection_in_pieces=True)
            self.mission = reg[0].lower()
            self.mission = self.mission.replace("one of", "")  # keep ambiguity
        return self._tokenize(self.mission)

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "IAMissionSpeaker requires kwargs"
        return cls(**kwargs)


class SilenceFeedbackIAMissionSpeaker(IAMissionSpeaker, SpeakerSpec):

    def __init__(self, **kwargs):
        super(SilenceFeedbackIAMissionSpeaker, self).__init__(**kwargs)

    def generate_feedback(self, gripper_pos: Tuple[int, int], return_string=False) -> (np.array, int):
        return self._tokenize(SILENCE), 0

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "SilenceFeedbackIAMissionSpeaker requires kwargs"
        return cls(**kwargs)


class MissionSpeaker(Speaker, SpeakerSpec):
    """ A speaker that mentions the shape and color of a piece (but provides no feedback)"""

    def generate_feedback(self, gripper_pos: Tuple[int, int], return_string=False) -> (np.array, int):
        raise NotImplementedError()

    def generate_mission(self) -> np.array:
        if self.mission is None:
            shape = self.task.target_piece.piece_config.shape
            color = self.task.target_piece.piece_config.color
            pos = self.task.target_piece.piece_config.rel_position
            self.mission = f"take the {color.value_name} {shape.value} at the {pos.value}".lower()
        return self._tokenize(self.mission)

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        return cls()
