from cogrip.tasks import TaskLoader
from neuact.agents.speakers.combined import PositivePieceFeedbackIAMissionSpeaker
from neuact.envs.manual_play import run_interactive_plot
from neuact.envs.registry import register_pento

if __name__ == '__main__':
    # set_random_seed(default_seed())
    # grid_config = GridConfig(20, 20, move_step=1, prevent_overlap=True)
    # task = Task.create_random(grid_config)
    # task_loader = TaskLoader([task])
    task_loader = TaskLoader.from_file("train", filter_map_size=20)
    env_name = register_pento(task_loader,
                              speaker_spec=PositivePieceFeedbackIAMissionSpeaker,
                              speaker_kwargs=dict(distance_threshold=3, time_threshold=6, preference_order="CSP"))
    run_interactive_plot(env_name, use_fov=True, show_fov=True)
