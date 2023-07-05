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

"""
    Evaluate the "feedback agents" on scenarios without feedback, except "yes this piece".
"""


def main():
    # model_name = agent.agent_name
    # results_path = default_results_dir(test_env_name, model_name)
    # split_name = "only_mission"
    split_name = "only_piece_feedback"
    print("split_name:", split_name)
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
    args = parser.parse_args()
    main()
