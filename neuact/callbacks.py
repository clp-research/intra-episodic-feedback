from collections import defaultdict

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback
from PIL import Image, ImageFont, ImageDraw
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv

from cogrip.env import EPISODE_OUTCOME_SUCCESS, CoGripEnv
from cogrip.language import trans_obs
from neuact import utils

font = ImageFont.load_default()
if utils.is_ubuntu():
    # with open("/usr/share/fonts/truetype/freefont/FreeMono.ttf", "rb") as f:
    #    font = ImageFont.truetype(font=BytesIO(f.read()), size=10)
    font = ImageFont.truetype("/usr/share/fonts/truetype/freefont/FreeMono.ttf", 10)
if utils.is_mac():
    font = ImageFont.truetype("Keyboard.ttf", 10)


def cache_image(gif_images, image, mission, success=None, upscale=20, feedback=None):
    bigger_img = image.repeat(upscale, axis=0).repeat(upscale, axis=1).astype(np.uint8)
    # draw mission statement
    if success is None:
        text_bg = (200, 200, 200)
    if success is True:
        text_bg = (0, 200, 0)
    if success is False:
        text_bg = (200, 0, 0)
    mission_img = Image.new("RGB", (bigger_img.shape[0], 20), text_bg)
    draw = ImageDraw.Draw(mission_img)
    draw.text((5, 5), mission, (0, 0, 0), font=font)
    mission_img = np.array(mission_img)

    feedback_img = Image.new("RGB", (bigger_img.shape[0], 20), (200, 200, 200))
    draw = ImageDraw.Draw(feedback_img)
    draw.text((5, 5), feedback if feedback else "", (0, 0, 0), font=font)
    feedback_img = np.array(feedback_img)
    image_with_text = np.concatenate([mission_img, feedback_img, bigger_img])

    gif_images.append(image_with_text)


class RecordVideoCallback(BaseCallback):

    def __init__(self, env, phase, n_freq=None, verbose=0):
        """
        :param env: to record the video for
        :param phase: train or eval
        :param n_freq: log video after each n training phases (ignored for eval)
        """
        super(RecordVideoCallback, self).__init__(verbose)
        self.env = env
        self.phase = phase
        self.n_freq = n_freq
        self.training_count = 0

    def _on_rollout_start(self):
        self.training_count += 1
        if self.n_freq:
            if self.n_freq == 1:
                self.log_video()
            elif self.training_count % self.n_freq != 0:
                self.log_video()

    def _on_step(self) -> bool:
        # this is called on eval; we don't want a video at each step
        if self.phase == "eval":
            self.log_video()
        return True

    def log_video(self):
        # print(f"Make {self.phase} video")
        gif_images = []
        policy = self.model
        env = self.env
        if isinstance(env, VecEnv):
            env = env.envs[0]
        obs = env.reset()
        mission, feedback, image = trans_obs(env, obs)
        cache_image(gif_images, image, mission)
        while True:
            action, _states = policy.predict(obs, deterministic=False if self.phase == "train" else True)
            obs, reward, done, info = env.step(action)
            mission, feedback, image = trans_obs(env, obs)
            if done:
                cache_image(gif_images, image, mission, info[EPISODE_OUTCOME_SUCCESS] == 1)
                break
            else:
                cache_image(gif_images, image, mission)
        env.reset()
        vid = torch.stack([torch.permute(torch.tensor(image), dims=(2, 0, 1)) for image in gif_images], dim=0)
        vid = torch.unsqueeze(vid, dim=0)
        tblogger = self.logger.output_formats[0]  # we simply assume that's the tb logger!
        tblogger.writer.add_video(f"{self.phase}/video", vid, global_step=self.num_timesteps)
        tblogger.writer.flush()


class SaveLatestCheckpointCallback(BaseCallback):

    def __init__(self, save_dir, only_on_training_end=False, verbose=0):
        """
        :param save_dir: the model is saved as model_latest.zip in save_path
        :param only_on_training_end: save only once after training (to improve training speed)
        """
        super(SaveLatestCheckpointCallback, self).__init__(verbose)
        self.save_dir = save_dir
        self.only_on_training_end = only_on_training_end

    def _on_step(self) -> bool:
        return True

    def _save_path(self):
        return f"{self.save_dir}/latest_model.zip"

    def _on_rollout_end(self):
        if self.only_on_training_end:
            return
        self.model.save(self._save_path())

    def _on_training_end(self) -> bool:
        self.model.save(self._save_path())
        return True


class LogEpisodicEnvInfoCallback(BaseCallback):
    """
        Log additional info accumulated by the monitor
    """

    def __init__(self, compute_std: bool = False, verbose=0):
        super(LogEpisodicEnvInfoCallback, self).__init__(verbose)
        self.compute_std = compute_std

    def _init_callback(self) -> None:
        if isinstance(self.training_env, DummyVecEnv):  # at training
            self.n_envs = self.training_env.num_envs
        elif isinstance(self.training_env, CoGripEnv):  # at eval
            self.n_envs = 1
        self.episode_records = [defaultdict(list) for _ in range(self.n_envs)]
        self.episode_means = [defaultdict(list) for _ in range(self.n_envs)]
        self.episode_totals = [defaultdict(list) for _ in range(self.n_envs)]
        self.n_steps = [0 for _ in range(self.n_envs)]
        self.n_episodes = [0 for _ in range(self.n_envs)]
        self.mode = "accumulate_per_rollout"  # or "moving_average" (keep history)

    def on_eval_step(self, _locals, _globals):
        if _locals["done"]:
            # env.reset has been already called: we need to get the infos from locals
            self._collect_steps([_locals["info"]])
        else:
            self.on_step()

    def _on_step(self) -> bool:
        if isinstance(self.training_env, DummyVecEnv):
            env_infos = self.training_env.buf_infos
        elif isinstance(self.training_env, CoGripEnv):
            env_infos = [self.training_env.info]
        return self._collect_steps(env_infos)

    def _collect_steps(self, env_infos):
        for idx, infos in enumerate(env_infos):
            self.n_steps[idx] += 1  # keep separate count b.c. we do not store zeros
            self._collect_step(idx, infos)
        return True

    def _collect_step(self, idx, infos):
        episode_records = self.episode_records[idx]
        is_done = False
        for k, step_value in infos.items():
            if k == "done":  # do not store
                is_done = step_value
            if k.startswith("episode/") or k.startswith("step/"):  # ours
                episode_records[k].append(step_value)

        for k in list(infos.keys()):
            if k.startswith("episode/") or k.startswith("step/"):  # ours
                del infos[k]  # consumed record value must be removed from stateful info (or set to zero)
        """ safety check
        speaker_steps = sum([len(episode_records[k]) for k in episode_records if k.startswith("step/speaker")])
        n_steps = self.n_steps[idx]
        if speaker_steps < n_steps:
            print()
        """
        if is_done:
            self.n_episodes[idx] += 1
            n_episodes = self.n_episodes[idx]
            episode_means = self.episode_means[idx]
            episode_totals = self.episode_totals[idx]
            # add zero counts for step events that never happened before, but did happen in this episode
            for k in episode_records:
                if k.startswith("step/") and k not in episode_means:
                    aggregated_values = episode_means[k]
                    while len(aggregated_values) < n_episodes - 1:  # pre-pend
                        episode_means[k].append(0.)
            # add values for step events that have happened this episode
            for k, values in episode_records.items():
                total, n_steps = sum(values), self.n_steps[idx]
                if k.startswith("step/"):  # a measure we want to know per step
                    episode_means[k].append(total / n_steps)
                if k.startswith("episode/"):  # a measure we want to know per episode
                    episode_totals[k].append(total)
            episode_records.clear()
            self.n_steps[idx] = 0
            # add zero counts for step events that were recorded before, but did not happen in this episode
            for k, values in episode_means.items():
                if k.startswith("step/"):
                    while len(values) < n_episodes:  # append
                        episode_means[k].append(0.)

    def _on_rollout_end(self) -> None:
        # accumulate record values over all envs
        all_recorded_values = defaultdict(list)
        for idx in range(self.n_envs):
            # clear intermediate records
            self.episode_records[idx].clear()
            self.n_steps[idx] = 0
            # collect step-actions over all episodes in the rollout
            for k, values in self.episode_means[idx].items():
                all_recorded_values[k].extend(values)
            # collect episode outcomes over all over episodes in the rollout
            for k, values in self.episode_totals[idx].items():
                all_recorded_values[k].extend(values)
        # micro-average record values over all environments
        for k, record_values in all_recorded_values.items():
            total, n_episodes = sum(record_values), len(record_values)
            average = total / n_episodes
            self.logger.record(k, average)
            if self.compute_std:
                std = np.std(record_values)
                self.logger.record(f"{k}/std", std)
        # clear history (not moving average)
        if self.mode == "accumulate_per_rollout":
            for idx in range(self.n_envs):
                self.n_episodes[idx] = 0
            for env_means in self.episode_means:
                env_means.clear()
            for env_totals in self.episode_totals:
                env_totals.clear()
