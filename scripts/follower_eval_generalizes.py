import argparse

from cogrip.tasks import TaskLoader
from neuact.agents.pento_follower import PentoFollower
from neuact.agents.speakers.combined import ThresholdedPieceFeedbackIAMissionSpeaker
from neuact.envs.follower_env import PentoFollowerEnv


def main(no_feedback):
    """
    Create a {split_name}.monitor.csv in the directory of the model
    """
    print("no_feedback:", no_feedback)
    if no_feedback:
        fb_dir = "nofb"
        model_path = "mi/pixels-we+lm-film/128-128-128/1024@8"
    else:
        fb_dir = "fb"
        model_path = "mifb/pixels-we+lm-film/128-128-128/1024@8"
    use_feedback = fb_dir == "fb"
    pref_order = "PCS"  # best
    model_dir = f"save_models/PentoEnv/{pref_order}/{fb_dir}/v1/{model_path}/"

    ckpt_path = f"{model_dir}/best_model.zip"
    agent = PentoFollower.create_large(use_mission=True, use_feedback=use_feedback)

    print(f"PO: {pref_order}")
    speaker_spec = ThresholdedPieceFeedbackIAMissionSpeaker
    speaker_kwargs = dict(distance_threshold=3, time_threshold=6, preference_order=pref_order.upper())

    # holdout generalization
    map_size = 20
    split_name = "holdout"
    # check both [4, 8]:
    num_pieces = [4, 8]
    task_loader = TaskLoader.from_file(split_name, filter_map_size=map_size,
                                       filter_num_pieces=num_pieces,
                                       verbose=True)
    env = PentoFollowerEnv.create(task_loader, return_feedback=use_feedback,
                                  speaker_spec=speaker_spec, speaker_kwargs=speaker_kwargs)

    agent.eval(env, results_dir=model_dir, ckpt_path=ckpt_path,
               model_name=f"{split_name}_{map_size}_4P_8P", n_episodes=len(task_loader))

    # map_size generalization
    map_size = 30
    split_name = "test"
    task_loader = TaskLoader.from_file(split_name, filter_map_size=map_size,
                                       filter_num_pieces=num_pieces,
                                       verbose=True)
    env = PentoFollowerEnv.create(task_loader, return_feedback=use_feedback,
                                  speaker_spec=speaker_spec, speaker_kwargs=speaker_kwargs)
    agent.eval(env, results_dir=model_dir, ckpt_path=ckpt_path,
               model_name=f"{split_name}_{map_size}_4P_8P", n_episodes=len(task_loader))

    num_pieces = [12, 18]
    task_loader = TaskLoader.from_file(split_name, filter_map_size=map_size,
                                       filter_num_pieces=num_pieces,
                                       verbose=True)
    env = PentoFollowerEnv.create(task_loader, return_feedback=use_feedback,
                                  speaker_spec=speaker_spec, speaker_kwargs=speaker_kwargs)
    agent.eval(env, results_dir=model_dir, ckpt_path=ckpt_path,
               model_name=f"{split_name}_{map_size}_12P_18P", n_episodes=len(task_loader))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-nofb", "--no_feedback", action="store_true")
    args = parser.parse_args()
    main(args.no_feedback)
