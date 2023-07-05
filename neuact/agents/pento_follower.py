from typing import Dict, Callable

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import configure_logger

from cogrip import registry
from neuact.callbacks import LogEpisodicEnvInfoCallback, RecordVideoCallback, SaveLatestCheckpointCallback
from neuact.agents.extractors.custom import CustomCombinedExtractor
from neuact.envs.follower_env import PentoFollowerEnv


def linear_schedule(initial_value: float, cap_value: float = None) -> Callable[[float], float]:
    if cap_value:
        print("Cap lr at", cap_value)
        assert initial_value >= cap_value, "cap_value should be smaller than or equal to initial_value"

    def func(progress_remaining: float) -> float:
        if progress_remaining < 0:  # sometimes this doesnt fit, so we cap it at 0
            print("progress_remaining must be in (1,...,0) but is", progress_remaining)
            progress_remaining = 0
        lr = progress_remaining * initial_value
        if cap_value and cap_value > lr:
            return cap_value
        return lr

    return func


class PentoFollower:

    def __init__(self, language_arch: str, vision_arch: str, policy_arch: Dict, fusion_arch: str,
                 vision_dims: int = 128, language_dims: int = 128,
                 use_feedback: bool = False, use_mission: bool = False,
                 num_epochs: int = 4, batch_size: int = 64, lr: float = 2.5e-4, cap_lr: float = 2.5e-5,
                 num_envs: int = 4, buffer_per_env: int = 128):
        """

        :param language_arch: [we+lm,oh+lm,sent-pre,sent-bow]
            - we+lm: learn word embeddings (+gru) during training
            - oh+lm: use one-hot word embeddings (+gru) during training
            - sent-bow: use one-hot sentence embeddings (+ffn to project up to language dims)
            - sent-pre: use pre-trained SentBERT embeddings (+proj down to language_dims)
        :param vision_arch: [pixels,symbols]
            - pixels: use NatureCNN
            - symbols: use bag-of-words image embeddings (+cnn)
        :param fusion_arch: [linear,film]
        """
        self.vision_arch = vision_arch
        self.language_arch = language_arch
        self.fusion_arch = fusion_arch
        model_name = ""
        if use_mission and use_feedback:
            model_name = model_name + "mifb/"
        elif use_mission:
            model_name = model_name + "mi/"
        elif use_feedback:
            model_name = model_name + "fb/"
        model_name = model_name + f"{vision_arch}-{language_arch}-{fusion_arch}"
        policy_dims = policy_arch["pi"][0]  # assume these are the same for now
        feature_dims = vision_dims  # assume these are the same for now
        model_name = model_name + f"/{feature_dims}-{policy_dims}-{policy_dims}"
        model_name = model_name + f"/{buffer_per_env}@{num_epochs}"
        self.agent_name = model_name
        self.num_envs = num_envs
        self.buffer_per_env = buffer_per_env  # buffer size e.g. 4 * 128 = 512
        self.batch_size = batch_size  # mini-batch size e.g. 512 / 64 = 8 mini-batches
        self.num_epochs = num_epochs  # numer of update epochs e.g. 4 * 8 -> seeing each buffer example 4 times
        self.policy_kwargs = {
            "features_extractor_class": CustomCombinedExtractor,
            "features_extractor_kwargs": {
                "vision_arch": vision_arch,
                "vision_dims": vision_dims,
                "language_arch": language_arch,
                "language_dims": language_dims,
                "use_feedback": use_feedback,
                "use_mission": use_mission,
                "fusion_arch": fusion_arch
            },
            "net_arch": [policy_arch],  # non-shared
            "normalize_images": True if vision_arch == "pixels" else False
            # "optimizer_kwargs": {"lr": 1e-4, "betas": (0.9, 0.999), "eps": 1e-5}
        }
        self.schedule = linear_schedule(lr, cap_lr)
        self.algorithm = None
        self.eval_env = None
        self.callbacks = list()

    def setup(self, env_name: str, tensorboard_log=None):
        print(f"Setup agent for {env_name}")
        envs = make_vec_env(env_name, n_envs=self.num_envs, wrapper_class=PentoFollowerEnv.wrap_partial)
        self.algorithm = PPO("MultiInputPolicy",
                             envs,
                             learning_rate=self.schedule,
                             policy_kwargs=self.policy_kwargs,
                             tensorboard_log=tensorboard_log,
                             verbose=0,
                             clip_range=.2,  # default
                             clip_range_vf=.2,
                             # from "Part 1 of 3 — PPO Implementation: 11 Core Implementation Details"
                             ent_coef=0.01,
                             # from "Part 1 of 3 — PPO Implementation: 11 Core Implementation Details"
                             vf_coef=0.5,  # default
                             max_grad_norm=0.5,  # default
                             target_kl=0.015,  # early stopping
                             n_steps=self.buffer_per_env,
                             batch_size=self.batch_size,
                             n_epochs=self.num_epochs,
                             seed=None  # default; set later on train
                             )
        return self

    def load(self, ckpt_path: str, env_name: str):
        print(f"Load agent for {env_name} from {ckpt_path}")
        envs = registry.make_envs(env_name, self.num_envs)
        """ Note: This seems to reset the Adam optimizer (the learning rate is again the init one)"""
        self.algorithm = PPO.load(ckpt_path, env=envs)
        return self

    def set_logger(self, log_dir):
        logger = configure_logger(tensorboard_log=log_dir, tb_log_name="PPO", reset_num_timesteps=True)
        self.algorithm.set_logger(logger)
        self.algorithm.policy.features_extractor.logger = logger
        return logger

    def add_loginfo_callback(self):
        self.callbacks.append(LogEpisodicEnvInfoCallback())
        return self

    def add_train_video_callback(self, env: PentoFollowerEnv, every_n_rollouts=100):
        self.callbacks.append(RecordVideoCallback(env, phase="train", n_freq=every_n_rollouts))
        return self

    def add_checkpoint_callback(self, save_dir, only_on_training_end=False):
        self.callbacks.append(SaveLatestCheckpointCallback(save_dir, only_on_training_end))
        return self

    def add_eval_callback(self, eval_env: PentoFollowerEnv, eval_video_env: PentoFollowerEnv, save_dir: str,
                          episodes_per_env=5, every_n_rollouts=1):
        # use a separate env for eval
        self.callbacks.append(EvalCallback(eval_env,
                                           eval_freq=self.buffer_per_env * every_n_rollouts,
                                           n_eval_episodes=episodes_per_env,
                                           best_model_save_path=save_dir, verbose=False,
                                           callback_on_new_best=RecordVideoCallback(eval_video_env, phase="eval")))
        return self

    def train(self, num_timesteps: int = 25600, seed=1):  # 512 * 50
        assert self.algorithm is not None, "Call setup() or load() before train()"
        print("Seed:", seed)
        print("Steps:", num_timesteps)
        self.algorithm.set_random_seed(seed)
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)
        self.algorithm.learn(total_timesteps=num_timesteps,
                             progress_bar=True,
                             log_interval=-1,
                             callback=list(self.callbacks))
        return self

    def eval(self, env, results_dir: str, ckpt_path=None, n_episodes=200, deterministic=True, model_name=None):
        env = Monitor(env, results_dir + "/" + model_name if model_name else "")
        if ckpt_path:
            policy = PPO.load(ckpt_path, env=env)
        else:
            policy = self.algorithm.policy
        print("Env:", env.name)
        print("Agent:", self.agent_name)
        print(f"Eval {model_name} for {n_episodes} episodes")
        mean_reward, std_reward = evaluate_policy(policy, env, n_eval_episodes=n_episodes, deterministic=deterministic)
        print(model_name if model_name else "", "results:", mean_reward, "avg.", std_reward, "std.")

    @classmethod
    def create_small(cls, use_mission, use_feedback):
        f = cls(language_arch="we+lm", vision_arch="pixels", fusion_arch="film", vision_dims=64, language_dims=64,
                policy_arch=dict(pi=[64, 64], vf=[64, 64]), num_envs=4, num_epochs=4, buffer_per_env=256,
                use_mission=use_mission, use_feedback=use_feedback)
        print("Create small agent", f.agent_name)
        return f

    @classmethod
    def create_large(cls, use_mission, use_feedback):
        f = cls(language_arch="we+lm", vision_arch="pixels", fusion_arch="film", vision_dims=128, language_dims=128,
                policy_arch=dict(pi=[128, 128], vf=[128, 128]), num_envs=4, num_epochs=8, buffer_per_env=1024,
                use_mission=use_mission, use_feedback=use_feedback)
        print("Create large agent", f.agent_name)
        return f
