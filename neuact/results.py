import json

import pandas

col_names = dict(r="rewards", l="steps", t="time")


def load_results(file_path: str) -> pandas.DataFrame:
    headers = []
    with open(file_path) as file_handler:
        first_line = file_handler.readline()
        assert first_line[0] == "#"
        header = json.loads(first_line[1:])
        data_frame = pandas.read_csv(file_handler, index_col=None)
        headers.append(header)
        data_frame["t"] += header["t_start"]
    data_frame.sort_values("t", inplace=True)
    data_frame.reset_index(inplace=True)
    data_frame["t"] -= min(header["t_start"] for header in headers)
    return data_frame


def calculate(results_path):
    df = load_results(results_path)
    print(f"total: {len(df)}")
    for col in df.columns:
        if col in col_names:
            col_name = col_names[col]
            if col_name == "time":
                continue
            if col_name == "steps":
                print(f"{col_name:<8} avg: {df[col].mean():>7.2f} std: {df[col].std(ddof=0):>7.2f}")
            if col_name == "rewards":
                """
                neg_r = df[df.r < 0].r
                print(f"  neg: {len(neg_r)}, mean: {neg_r.mean():>7.2f}")
                pos_r = df[df.r > 0].r
                print(f"  pos: {len(pos_r)}, mean: {pos_r.mean():>7.2f}")
                """
                pos_r = df[df.r > 0].r
                print(f"  mSR: {(len(pos_r) / len(df[col])) * 100:>7.2f}")
                print(f"{col_name:<8} avg: {df[col].mean():>7.2f} std: {df[col].std(ddof=0):>7.2f}")
