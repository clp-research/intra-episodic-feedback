import math
from abc import ABC
from typing import Tuple

import numpy as np

from cogrip.pentomino.objects import Piece
from neuact.agents.speakers import Speaker, SILENCE, SpeakerSpec


class ThresholdedGripperAwareFeedbackSpeaker(Speaker, ABC):
    """
    A speaker that can "see" the gripper position and react to the changes of the gripper position.
    """

    def __init__(self, distance_threshold: int, time_threshold: int, **kwargs):
        """
        :param distance_threshold: the threshold for "essential" movement
        :param time_threshold: the number of steps where no "essential" movement happened
        """
        # (a) a distance-determined (cross a distance of 3 compared to the last gripper position)
        self.last_feedback_distance_threshold = distance_threshold
        # (b) a time-determined (e.g. waiting for 5 seconds)
        self.last_feedback_time_threshold = time_threshold  # actions
        self.wait_counter = 0
        super(ThresholdedGripperAwareFeedbackSpeaker, self).__init__(**kwargs)

    def generate_feedback(self, gripper_pos: Tuple[int, int], return_string=False) -> (np.array, int):
        if self._is_gripper_over_piece(gripper_pos):
            # print("gripper_over_piece")
            feedback, signal = self._generate_piece_feedback(gripper_pos)
        else:  # gripper is moving
            if self._is_distance_threshold_reached(gripper_pos):  # movement is essential
                # print("distance_threshold_reached")
                feedback, signal = self._generate_move_feedback(gripper_pos)
                # mark reference point for next threshold comparison
                self._mark_reference_point(gripper_pos)
            else:  # gripper is waiting
                # print("gripper is waiting")
                feedback, signal = self._generate_wait_feedback(gripper_pos)
        if feedback:  # whenever something has been said, we mark it
            self._mark_reference_point(gripper_pos)
        if return_string:
            return feedback, signal
        return self._tokenize(feedback), signal

    def _mark_reference_point(self, gripper_pos):
        self.last_gripper_pos = gripper_pos
        self.wait_counter = 0  # reset wait counter

    def _generate_wait_feedback(self, gripper_pos):
        self.wait_counter += 1
        if self._is_time_threshold_reached():
            # print("_time_threshold_reached")
            # mark reference point, so that the wait action not immediately again
            self._mark_reference_point(gripper_pos)
            feedback, signal = self._on_gripper_waits()
        else:
            feedback, signal = SILENCE, 0
        return feedback, signal

    def _generate_move_feedback(self, gripper_pos):
        target_pos = self.task.target_piece.get_centroid()
        last_distance_to_target = math.dist(self.last_gripper_pos, target_pos)
        current_distance_to_target = math.dist(gripper_pos, target_pos)
        if current_distance_to_target < last_distance_to_target:
            feedback, signal = self._on_gripper_moves_towards_target()
        else:
            feedback, signal = self._on_gripper_moves_away_from_target()
        return feedback, signal

    def _generate_piece_feedback(self, gripper_pos):
        underlying_piece = self._get_underlying_piece(gripper_pos)
        if self.task.target_piece.id_n == underlying_piece.id_n:
            feedback, signal = self._on_gripper_over_target()
        else:
            feedback, signal = self._on_gripper_over_other()
        return feedback, signal

    def _is_time_threshold_reached(self):
        # print(self.wait_counter)
        return self.wait_counter >= self.last_feedback_time_threshold

    def _is_distance_threshold_reached(self, gripper_pos) -> bool:
        distance_crossed_since_last_feedback = math.dist(self.last_gripper_pos, gripper_pos)
        # print(distance_crossed_since_last_feedback)
        return distance_crossed_since_last_feedback >= self.last_feedback_distance_threshold

    def _get_underlying_piece(self, gripper_pos) -> Piece:
        gripper_tile = self.board.get_tile(*gripper_pos)
        piece = gripper_tile.objects[-1]
        return piece

    def _is_gripper_over_piece(self, gripper_pos) -> bool:
        gripper_tile = self.board.get_tile(*gripper_pos)
        return len(gripper_tile.objects) > 0

    def _on_gripper_over_target(self) -> (str, int):
        raise NotImplementedError()

    def _on_gripper_over_other(self) -> (str, int):
        raise NotImplementedError()

    def _on_gripper_waits(self) -> (str, int):
        raise NotImplementedError()

    def _on_gripper_moves_towards_target(self) -> (str, int):
        raise NotImplementedError()

    def _on_gripper_moves_away_from_target(self) -> (str, int):
        raise NotImplementedError()


class PieceFeedbackSpeaker(ThresholdedGripperAwareFeedbackSpeaker, SpeakerSpec):
    """ A speaker that says yes or no when the gripper is over a piece """

    def __init__(self, **kwargs):
        # The follower should learn sentence starting with "not ..." is negative and "yes ..." is positive.
        # Note: We add "this piece" and "take this piece" to distinguish later for directional feedback like
        # "yes this direction". Otherwise, the follower might try to perform the grip action always when
        # "yes ..." is given as a feedback, but here it needs to distinguish positive grip and movement actions.
        self.negative_feedback = "not this piece"
        self.positive_feedback = "yes this piece"
        super(PieceFeedbackSpeaker, self).__init__(**kwargs)

    def _on_gripper_over_target(self) -> (str, int):
        return self.positive_feedback, 1

    def _on_gripper_over_other(self) -> (str, int):
        return self.negative_feedback, -1

    def _on_gripper_waits(self) -> (str, int):
        return SILENCE, 0

    def _on_gripper_moves_towards_target(self) -> (str, int):
        return SILENCE, 0

    def _on_gripper_moves_away_from_target(self) -> (str, int):
        return SILENCE, 0

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "PieceFeedbackSpeaker requires kwargs"
        return cls(**kwargs)


class ThresholdedPieceFeedbackSpeaker(PieceFeedbackSpeaker, SpeakerSpec):
    """ A speaker that provides feedback at each step (but no mission) """

    def __init__(self, **kwargs):
        self.negative_step_feedback = "not this direction"
        self.positive_step_feedback = "yes this direction"
        super(ThresholdedPieceFeedbackSpeaker, self).__init__(**kwargs)

    def _on_gripper_waits(self) -> (str, int):
        return self.mission, 0

    def _on_gripper_moves_towards_target(self) -> (str, int):
        return self.positive_step_feedback, 1

    def _on_gripper_moves_away_from_target(self) -> (str, int):
        return self.negative_step_feedback, -1

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "ThresholdedPieceFeedbackSpeaker requires kwargs"
        return cls(**kwargs)


class ThresholdedPieceFeedbackSilenceMissionSpeaker(ThresholdedPieceFeedbackSpeaker, SpeakerSpec):
    def __init__(self, distance_threshold: int, time_threshold: int, **kwargs):
        super(ThresholdedPieceFeedbackSilenceMissionSpeaker, self).__init__(distance_threshold=distance_threshold,
                                                                            time_threshold=time_threshold)

    def generate_mission(self) -> np.array:
        self.mission = SILENCE
        return self._tokenize(self.mission)

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "ThresholdedPieceFeedbackSilenceMissionSpeaker requires kwargs"
        return cls(**kwargs)


class ThresholdedPieceFeedbackUnderspecificMissionSpeaker(ThresholdedPieceFeedbackSpeaker, SpeakerSpec):
    def __init__(self, distance_threshold: int, time_threshold: int, **kwargs):
        super(ThresholdedPieceFeedbackUnderspecificMissionSpeaker, self).__init__(distance_threshold=distance_threshold,
                                                                                  time_threshold=time_threshold)

    def generate_mission(self) -> np.array:
        self.mission = "take the piece"
        return self._tokenize(self.mission)

    @classmethod
    def create(cls, kwargs=None) -> Speaker:
        assert kwargs is not None, "ThresholdedPieceFeedbackUnderspecificMissionSpeaker requires kwargs"
        return cls(**kwargs)
