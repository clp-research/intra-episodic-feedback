import argparse

from neuact.results import calculate

"""
pref. Order | mSR Val | w\ FB | mSR Test | w\ FB
CSP				
CPS				
PSC				
PCS				
SPC				
SCP				
"""


def main(split_name, no_feedback):
    # model_name = agent.agent_name
    # results_path = default_results_dir(test_env_name, model_name)
    print("split_name:", split_name)
    print("no_feedback:", no_feedback)
    if no_feedback:
        fb_dir = "nofb"
        model_dir = "mi/pixels-we+lm-film/128-128-128/1024@8"
    else:
        fb_dir = "fb"
        model_dir = "mifb/pixels-we+lm-film/128-128-128/1024@8"
    preference_orders = ["CPS", "CSP", "PCS", "PSC", "SCP", "SPC"]
    for po in preference_orders:
        print("PO:", po)
        print("--------")
        results_path = f"results/PentoEnv/{po}/{fb_dir}/v1/{model_dir}/{split_name}.monitor.csv"
        calculate(results_path)
        print("-" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("split_name", type=str, help="[val,test,holdout]")
    parser.add_argument("-nofb", "--no_feedback", action="store_true")
    args = parser.parse_args()
    main(args.split_name, args.no_feedback)
