import argparse

from neuact.results import calculate

"""
pref. Order	PCS	PCS \w FB
mSR ho 20x20 4P		
mSR ho 20x20 8P		
mSR test 30x30 4P		
mSR test 30x30 8P		
mSR test 30x30 12P		
mSR test 30x30 18P					
"""


def main(no_feedback):
    # model_name = agent.agent_name
    # results_path = default_results_dir(test_env_name, model_name)
    print("no_feedback:", no_feedback)
    if no_feedback:
        fb_dir = "nofb"
        model_dir = "mi/pixels-we+lm-film/128-128-128/1024@8"
    else:
        fb_dir = "fb"
        model_dir = "mifb/pixels-we+lm-film/128-128-128/1024@8"
    po = "PCS"
    print("PO:", po)
    print("--------")
    split_name = "holdout"
    map_size = 20
    file_name = f"{split_name}_{map_size}_4P_8P"
    print("Results", file_name)
    results_path = f"results/PentoEnv/{po}/{fb_dir}/v1/{model_dir}/{file_name}.monitor.csv"
    calculate(results_path)
    print("-" * 60)

    split_name = "test"
    map_size = 30
    file_name = f"{split_name}_{map_size}_4P_8P"
    print("Results", file_name)
    results_path = f"results/PentoEnv/{po}/{fb_dir}/v1/{model_dir}/{file_name}.monitor.csv"
    calculate(results_path)
    print("-" * 60)

    file_name = f"{split_name}_{map_size}_12P_18P"
    print("Results", file_name)
    results_path = f"results/PentoEnv/{po}/{fb_dir}/v1/{model_dir}/{file_name}.monitor.csv"
    calculate(results_path)
    print("-" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-nofb", "--no_feedback", action="store_true")
    args = parser.parse_args()
    main(args.no_feedback)
