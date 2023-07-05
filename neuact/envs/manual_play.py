import gym
import numpy as np

from cogrip.language import decode_sent
from cogrip.wrappers import PartialVisionObsWrapper, FullyPartialObsWrapper
from neuact.envs import Actions


def run_interactive_plot(env_name, use_fov=False, show_fov=False):
    from matplotlib import pyplot as plt

    def trans_obs(env, obs):
        if "mission" in obs:
            mission = decode_sent(obs["mission"])
        else:
            mission = "<no mission>"

        if "feedback" in obs:
            feedback = decode_sent(obs["feedback"])
        else:
            feedback = "<no feedback>"

        image = env.render(mode="rgb_array")
        return mission, feedback, image

    def prep_ax(ax, mission, feedback, image, step, reward, total_reward, context=None):
        if context is None:
            ax.axes.xaxis.set_ticklabels([])
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticklabels([])
            ax.axes.yaxis.set_ticks([])
        ax.set_title(
            f"mi: {mission}\nfb: {feedback}\nstep: {step} reward: {np.around(reward, 2)} total: {np.around(total_reward, 2)}",
            loc='left')
        if context:
            context.set_data(image)
        else:
            context = ax.imshow(image)
        return context

    global context
    global steps
    global total_reward
    steps = 0
    total_reward = 0
    print("Load env:", env_name)
    env = gym.make(env_name)
    if use_fov:
        env = PartialVisionObsWrapper(env)
        if show_fov:
            env = FullyPartialObsWrapper(env)
    obs = env.reset()

    fig, ax = plt.subplots(1, 1)
    mission, feedback, image = trans_obs(env, obs)
    context = prep_ax(ax, mission, feedback, image, step=0, reward=0, total_reward=0)
    fig.canvas.draw_idle()
    plt.pause(.1)

    def on_press(event):
        key = event.key.lower()
        action = Actions.wait
        if key == "left":
            action = Actions.left
        if key == "right":
            action = Actions.right
        if key == "up":
            action = Actions.up
        if key == "down":
            action = Actions.down
        if key == " ":
            action = Actions.grip
        obs, reward, done, _ = env.step(action)
        global context, steps, total_reward
        steps += 1
        total_reward += reward
        mission, feedback, image = trans_obs(env, obs)
        context = prep_ax(ax, mission, feedback, image, step=steps, reward=reward, total_reward=total_reward,
                          context=context)
        if done:
            env.reset()
            steps = 0
            total_reward = 0
        plt.pause(.1)

    fig.canvas.mpl_connect('key_press_event', on_press)
    plt.show()
