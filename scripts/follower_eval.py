import argparse

from cogrip.tasks import TaskLoader
from neuact.agents.pento_follower import PentoFollower
from neuact.agents.speakers.combined import ThresholdedPieceFeedbackIAMissionSpeaker
from neuact.envs.follower_env import PentoFollowerEnv


def main(split_name, map_size, no_feedback, num_pieces=None):
    """
    Create a {split_name}.monitor.csv in the directory of the model
    """
    print("split_name:", split_name)
    print("no_feedback:", no_feedback)
    print("filter_map_size:", map_size)
    print("filter_num_pieces:", num_pieces)
    if no_feedback:
        fb_dir = "nofb"
        model_path = "mi/pixels-we+lm-film/128-128-128/1024@8"
    else:
        fb_dir = "fb"
        model_path = "mifb/pixels-we+lm-film/128-128-128/1024@8"
    use_feedback = fb_dir == "fb"
    preference_orders = ["CSP", "CPS", "PSC", "PCS", "SPC", "SCP"]
    for pref_order in preference_orders:
        model_dir = f"save_models/PentoEnv/{pref_order}/{fb_dir}/v1/{model_path}/"

        speaker_spec = ThresholdedPieceFeedbackIAMissionSpeaker
        speaker_kwargs = dict(distance_threshold=3, time_threshold=6, preference_order=pref_order.upper())

        task_loader = TaskLoader.from_file(split_name, filter_map_size=map_size, filter_num_pieces=num_pieces)
        env = PentoFollowerEnv.create(task_loader, return_feedback=use_feedback,
                                      speaker_spec=speaker_spec, speaker_kwargs=speaker_kwargs)

        ckpt_path = f"{model_dir}/best_model.zip"
        agent = PentoFollower.create_large(use_mission=True, use_feedback=use_feedback)

        print(f"PO: {pref_order}")
        num_tasks = len(task_loader)
        agent.eval(env, results_dir=model_dir, ckpt_path=ckpt_path, model_name=split_name, n_episodes=num_tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("split_name", type=str, help="[val,test,holdout]")
    parser.add_argument("map_size", type=int, help="[20,30]", default=20)
    parser.add_argument("-n", "--num_pieces", type=int, help="[4,8,12,18]", default=None)
    parser.add_argument("-nofb", "--no_feedback", action="store_true")
    args = parser.parse_args()
    main(args.split_name, args.map_size, args.no_feedback, args.num_pieces)
