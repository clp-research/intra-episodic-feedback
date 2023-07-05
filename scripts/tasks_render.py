import math

from matplotlib import pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import set_random_seed

from cogrip.language import decode_sent
from cogrip.tasks import TaskLoader
from neuact.agents.speakers.combined import ThresholdedPieceFeedbackIAMissionSpeaker
from neuact.envs.follower_env import PentoFollowerEnv
from neuact.envs.registry import register_pento
from neuact.utils import default_seed


def prep_ax(ax, image, mission, feedback):
    ax.axes.xaxis.set_ticklabels([])
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticklabels([])
    ax.axes.yaxis.set_ticks([])
    ax.set_title("mi: " + mission + "\nfb: " + feedback, loc='left')
    ax.imshow(image)


if __name__ == '__main__':
    set_random_seed(default_seed())
    """
        We can simply use the same task loader N times, because the reset is a loop in DummyVecEnv 
        
        def reset(self) -> VecEnvObs:
            for env_idx in range(self.num_envs):
                obs = self.envs[env_idx].reset()
                self._save_obs(env_idx, obs)
            return self._obs_from_buf()
    """
    task_loader = TaskLoader.from_file("train", filter_map_size=20, do_shuffle=True)
    env_name = register_pento(task_loader,
                              speaker_spec=ThresholdedPieceFeedbackIAMissionSpeaker,
                              speaker_kwargs=dict(distance_threshold=3, time_threshold=6, preference_order="PCS"))
    n_envs = 16
    fig_size = (8, 6)
    if n_envs > 4:
        fig_size = (18, 14)
    env = make_vec_env(env_name, n_envs=n_envs, wrapper_class=PentoFollowerEnv.wrap_fully_partial)
    # env = make_vec_env(env_name, n_envs=n_envs, wrapper_class=PentoFollowerEnv.wrap_partial)
    obs = env.reset()
    if "mission" in obs:
        mission = [decode_sent(m) for m in obs["mission"]]
    else:
        mission = ["<no mission>" for _ in range(n_envs)]

    if "feedback" in obs:
        feedback = [decode_sent(m) for m in obs["feedback"]]
    else:
        feedback = ["<no feedback>" for _ in range(n_envs)]
    # prepare plot
    w = int(n_envs / int(math.sqrt(n_envs)))
    fig, ax = plt.subplots(w, w, figsize=fig_size)
    tuples = [(r, c) for c in range(w) for r in range(w)]
    images = env.get_images()
    for idx, t in enumerate(tuples):
        prep_ax(ax[t], images[idx], mission[idx], feedback[idx])
    # plt.tight_layout()
    plt.show()
