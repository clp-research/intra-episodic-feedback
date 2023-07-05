import argparse

import numpy as np
from stable_baselines3 import PPO

from cogrip.constants import COLORS_6, SHAPES_9
from cogrip.core.grid import GridConfig
from cogrip.language import trans_obs
from cogrip.tasks import TaskLoader, Task
from cogrip.wrappers import PartialVisionObsWrapper, FullyPartialObsWrapper
from neuact import utils
from neuact.callbacks import cache_image
from neuact.agents.speakers.combined import ThresholdedPieceFeedbackIAMissionSpeaker
from PIL import ImageFont

from neuact.envs.follower_env import PentoFollowerEnv

font = ImageFont.load_default()
if utils.is_mac():
    font = ImageFont.truetype("Keyboard.ttf", 10)


def prep_ax(ax, mission, feedback, image, step, reward, total_reward, context=None, done=None):
    if context is None:
        ax.axes.xaxis.set_ticklabels([])
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticks([])
    title_msg = ""
    title_msg += f"mi: {mission}\n"
    title_msg += f"fb: {feedback}\n"
    title_msg += f"step: {step} reward: {np.around(reward, 2)} total: {np.around(total_reward, 2)}\n"
    if done is not None:
        title_msg += "SUCCESS" if done else "FAILURE"
    ax.set_title(title_msg, loc='left')
    if context:
        context.set_data(image)
    else:
        context = ax.imshow(image)
    return context


def test(ckpt_path: str, pref_order: str, split_name: str, map_size, num_pieces,
         num_frames, step_time=.2, done_time=1, deterministic=True,
         do_plot=False, do_gif=False):
    print("Load model from", ckpt_path)
    use_feedback = True
    if split_name is None:
        grid_config = GridConfig(20, 20, 1, True)
        colors = COLORS_6
        shapes = SHAPES_9
        task_loader = TaskLoader([Task.create_random(grid_config, num_pieces=4, colors=colors, shapes=shapes)
                                  for _ in range(10)])
    else:
        task_loader = TaskLoader.from_file(split_name, filter_map_size=map_size, filter_num_pieces=num_pieces,
                                           force_shuffle=True)
    fb_type = "full"
    speaker_spec = ThresholdedPieceFeedbackIAMissionSpeaker
    # speaker_spec = PositivePieceFeedbackIAMissionSpeaker
    speaker_kwargs = dict(distance_threshold=3, time_threshold=6, preference_order=pref_order.upper())
    env = PentoFollowerEnv.create(task_loader, return_feedback=use_feedback,
                                  speaker_spec=speaker_spec, speaker_kwargs=speaker_kwargs)
    fov_env = PartialVisionObsWrapper(env)
    overview_env = FullyPartialObsWrapper(fov_env)
    policy = PPO.load(ckpt_path, env=fov_env)

    # policy.set_random_seed(np.random.random_integers(1, 1000000))
    # model.set_parameters(file_name)
    frame_count = 0
    if do_gif:
        gif_images = []
        gif_fov_images = []
    obs = overview_env.reset()
    # prepare plot
    steps = 0
    total_rewards = 0
    if do_plot:
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))
        mission, feedback, image = trans_obs(overview_env, obs)
        context = prep_ax(ax, mission, feedback, image, step=0, reward=0, total_reward=0)
    wait_count = 0
    wait_threshold = 5
    while frame_count < num_frames:
        while True:
            action, _states = policy.predict(obs, deterministic=deterministic)
            # print(action)
            obs, reward, done, info = overview_env.step(action)
            mission, feedback, image = trans_obs(overview_env, obs)
            steps += 1
            frame_count += 1
            total_rewards += reward
            # print(f"\rMission: {mission} Reward: {reward} Step: {steps} Frame: {frame_count} Feedback: {feedback}",
            #      end="", flush=True)
            if do_plot:
                context = prep_ax(ax, mission, feedback, image, step=steps, reward=reward, total_reward=total_rewards,
                                  context=context, done=info["episode/success"] if done else None)
                fig.canvas.draw_idle()
                plt.pause(step_time)
            if "step/action/grip" in info:
                wait_count += 1
                print("Grip")
            if "step/action/wait" in info:
                wait_count += 1
                print("Wait")
            if do_gif and not done:
                cache_image(gif_images, image, mission,
                            feedback=feedback)
                mission, feedback, fov_image = trans_obs(fov_env, obs)
                cache_image(gif_fov_images, fov_image, mission, upscale=30,
                            feedback=feedback)
            if wait_threshold < wait_count:
                done = True
                info["episode/success"] = 0
            if done:
                wait_count = 0
                if do_gif:
                    cache_image(gif_images, image, mission, info["episode/success"] == 1,
                                feedback=feedback)
                    mission, feedback, fov_image = trans_obs(fov_env, obs)
                    cache_image(gif_fov_images, fov_image, mission, info["episode/success"] == 1, upscale=30,
                                feedback=feedback)
                steps = 0
                total_rewards = 0
                obs = overview_env.reset()
                if do_plot:
                    mission, feedback, image = trans_obs(overview_env, obs)
                    plt.pause(done_time)
                    context = prep_ax(ax, mission, feedback, image, step=0, reward=0, total_reward=0, context=context)
                break
    if do_gif:
        import imageio
        print(f"Save gif with {len(gif_images)} frames")
        imageio.mimsave(f'gifs/{pref_order}_{fb_type}_model_overview.gif', gif_images, fps=4, loop=0)
        imageio.mimsave(f'gifs/{pref_order}_{fb_type}_model_fov.gif', gif_fov_images, fps=4, loop=0)


def main(ckpt_path, pref_order, step_time, done_time):
    assert utils.is_mac(), "Only run this test script locally!"

    do_gif = True
    do_plot = False
    num_frames = 200

    split_name = "val"
    task_map_size = 20
    task_num_pieces = [4, 8]

    test(ckpt_path, pref_order=pref_order, split_name=split_name, map_size=task_map_size, num_pieces=task_num_pieces,
         num_frames=num_frames, step_time=step_time, done_time=done_time, deterministic=True,
         do_plot=do_plot, do_gif=do_gif)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    pref_order = "SCP"
    fb_dir = "fb"
    ckpt_path = f"save_models/PentoEnv/{pref_order}/{fb_dir}/v1/mifb/pixels-we+lm-film/128-128-128/1024@8/best_model.zip"
    step_time = .1
    done_time = 2
    main(ckpt_path, pref_order, step_time, done_time)
