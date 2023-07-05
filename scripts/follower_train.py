import argparse

import torch.cuda
from stable_baselines3.common.logger import HParam
from stable_baselines3.common.utils import set_random_seed

from cogrip.tasks import TaskLoader
from neuact import utils
from neuact.agents.pento_follower import PentoFollower
from neuact.agents.speakers.combined import ThresholdedPieceFeedbackIAMissionSpeaker
from neuact.envs.follower_env import PentoFollowerEnv
from neuact.utils import default_log_dir, default_save_dir, default_seed
from neuact.envs.registry import register_pento

if utils.is_ubuntu():
    torch.cuda.set_per_process_memory_fraction(0.2)


def main(model_size, no_feedback, pref_order, dry_run):
    seed = default_seed()
    set_random_seed(seed)

    speaker_spec = ThresholdedPieceFeedbackIAMissionSpeaker
    speaker_kwargs = dict(distance_threshold=3, time_threshold=6, preference_order=pref_order.upper())
    use_feedback = not no_feedback
    use_mission = True
    grid_size = 20
    hparams = dict(speaker=pref_order, use_feedback=use_feedback, use_mission=use_mission, env_grid_size=grid_size)

    def create_env(task_loader):
        return PentoFollowerEnv.create(task_loader, return_feedback=use_feedback,
                                       speaker_spec=speaker_spec, speaker_kwargs=speaker_kwargs)

    loaders, splits = TaskLoader.all_from_file(filter_map_size=grid_size, do_shuffle=True)
    env_name = register_pento(loaders["train"], return_feedback=use_feedback,
                              speaker_spec=speaker_spec, speaker_kwargs=speaker_kwargs)
    if model_size == "small":
        agent = PentoFollower.create_small(use_mission=use_mission, use_feedback=use_feedback)
    elif model_size == "large":
        agent = PentoFollower.create_large(use_mission=use_mission, use_feedback=use_feedback)
    else:
        raise ValueError(model_size)
    timestamp = str(utils.get_current_timestamp())[-5:]  # to distinguish runs with the same configuration
    print("Timestamp:", timestamp)
    save_dir = default_save_dir(env_name, agent.agent_name, sub_dir=timestamp)
    log_dir = default_log_dir(env_name, agent.agent_name, sub_dir=timestamp)
    agent.setup(env_name)  # uses vec env which needs a registered name

    if utils.is_ubuntu():
        agent.add_loginfo_callback()
        every_n_rollouts = 50
        if dry_run:
            every_n_rollouts = 1

        # add video callback (must use a different task loader)
        agent.add_train_video_callback(create_env(loaders["train"].clone()), every_n_rollouts=every_n_rollouts)

        # add eval callback
        agent.add_eval_callback(create_env(loaders["val"]), create_env(loaders["val"].clone()), save_dir,
                                episodes_per_env=len(loaders["val"]), every_n_rollouts=every_n_rollouts)

        logger = agent.set_logger(log_dir)
        logger.record("hparams", HParam(hparam_dict=hparams, metric_dict={"eval/mean_reward": 0, "episode/success": 0}))

    # small: num_timesteps=256000 * 4 then buffer_per_env 256 -> 1000 training phases for 4 envs
    # large: num_timesteps=256000 * 4 then buffer_per_env 1024 -> 250 training phases for 4 envs
    # lets do 10M
    time_steps = 2_560_000 * 4
    if dry_run:
        time_steps = 100000 * 4
    agent.train(num_timesteps=time_steps, seed=seed)

    print("Evaluate on test")
    num_tasks = len(test_loader)
    if dry_run:
        num_tasks = 1
    agent.eval(create_env(test_loader), results_dir=save_dir, model_name="test", n_episodes=num_tasks)

    print("Evaluate on holdout")
    num_tasks = len(test_loader)
    if dry_run:
        num_tasks = 1
    agent.eval(create_env(holdout_loader), results_dir=save_dir, model_name="holdout", n_episodes=num_tasks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model_size", type=str, help="[small,large]", default="large")
    parser.add_argument("pref_order", type=str, help="[CSP,CPS,PSC,PCS,SPC,SCP]", default="CSP")
    parser.add_argument("-nofb", "--no_feedback", action="store_true")
    parser.add_argument("-d", "--dry_run", action="store_true")
    args = parser.parse_args()
    main(args.model_size, args.no_feedback, args.pref_order, args.dry_run)
