import argparse

from cogrip.tasks import TaskLoader
from neuact.agents.pento_follower import PentoFollower
from neuact.agents.speakers.combined import PositivePieceFeedbackIAMissionSpeaker, PieceFeedbackIAMissionSpeaker
from neuact.agents.speakers.feedback import ThresholdedPieceFeedbackSilenceMissionSpeaker
from neuact.envs.follower_env import PentoFollowerEnv

"""
    We take the trained models (mi+fb) for each preference order and test under the following conditions:
    (a) only given the mission statement
    (b) only given the feedback signal
    (c) both (we already have these results) 
"""


def main():
    """
    Create a {split_name}.monitor.csv in the directory of the model
    """
    fb_dir = "fb"
    model_path = "mifb/pixels-we+lm-film/128-128-128/1024@8"
    preference_orders = ["CSP", "CPS", "PSC", "PCS", "SPC", "SCP"]
    for pref_order in preference_orders:
        print(f"PO: {pref_order}")
        model_dir = f"save_models/PentoEnv/{pref_order}/{fb_dir}/v1/{model_path}/"
        ckpt_path = f"{model_dir}/best_model.zip"
        agent = PentoFollower.create_large(use_mission=True, use_feedback=True)

        for condition in ["only_piece_feedback"]:
            print(f"Condition: {condition}")
            if condition == "only_feedback":
                speaker_spec = ThresholdedPieceFeedbackSilenceMissionSpeaker
                speaker_kwargs = dict(distance_threshold=3, time_threshold=6, preference_order=pref_order.upper())
            if condition == "only_mission":
                speaker_spec = PositivePieceFeedbackIAMissionSpeaker
                speaker_kwargs = dict(distance_threshold=3, time_threshold=6, preference_order=pref_order.upper())
            if condition == "only_piece_feedback":
                speaker_spec = PieceFeedbackIAMissionSpeaker
                speaker_kwargs = dict(distance_threshold=3, time_threshold=6, preference_order=pref_order.upper())
            task_loader = TaskLoader.from_file("test", filter_map_size=20)
            env = PentoFollowerEnv.create(task_loader, return_feedback=True,
                                          speaker_spec=speaker_spec, speaker_kwargs=speaker_kwargs)
            agent.eval(env, results_dir=model_dir, ckpt_path=ckpt_path, model_name=condition,
                       n_episodes=len(task_loader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main()
