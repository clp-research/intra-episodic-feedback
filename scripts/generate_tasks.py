import json

from cogrip.constants import COLORS_6, SHAPES_9
from cogrip.pentomino.symbolic.types import SymbolicPiece
from cogrip.tasks import Task, store_tasks_to_json


def load_splits_from_json(file_name="splits.json"):
    with open(file_name) as f:
        data = json.load(f)
    splits = dict()
    for split_name, symbolic_pieces in data.items():
        splits[split_name] = [SymbolicPiece.from_json(sp) for sp in symbolic_pieces]
    return splits


"""
    We want to generate the actual task given the possible target pieces (splits.json).
    A task is determined by the the target piece and the other pieces (distractors)
    on a board. There can be different sizes of boards and different numbers of pieces.
    
    When we assume 1,000,000 time steps and a maximum of 100 time steps per task, 
    then we can attempt 10,000 task (if the agent learns nothing). 
    
    In addition we want to produce tasks for two different map sizes (20,30)
    and different number of distractors (4,8,12,18) which leads to 6 variants.
    
    We would like to produce the same amount of tasks for the map sizes,
    lets say 3300 each. 
    
    This means for training that:
    
    - for 20x20: there are 2 variants (4,8); we produce 6 tasks for each
    target piece and variant which leads to 6*2*275=3300
    - for 30x30: there are 4 variants (4,8,12,18); we produce 3 tasks for
    each target piece and variant which leads to 3*4*275=3300
    
    Similarly we produce the validation and testing which leads to:
    
    - for 20x20: 6*2*25=300 validation tasks; 6*2*60=720 test tasks
    - for 30x30: 3*4*25=300 validation tasks; 3*4*60=720 test tasks
    
    As a result we store a structure like the following:
    {
        "train": List[Tasks],  -> 6600 (275 symbols)
        "val": List[Tasks],    ->  600 (25 symbols)
        "test": List[Tasks],   -> 1440 (60 symbols)
        "holdout": List[Tasks] -> 1728 (72 symbols)
   }     
"""


def main():
    splits = load_splits_from_json()

    for n, v in splits.items():
        print(n, len(v))

    # compute training tasks definitions
    sizes_and_pieces = {20: (6, (4, 8)), 30: (3, (4, 8, 12, 18))}
    colors = COLORS_6
    shapes = SHAPES_9
    tasks = dict()
    for split_name, target_pieces in splits.items():
        print("Generate for", split_name)
        tasks[split_name] = []
        for map_size, (num_tasks, variants) in sizes_and_pieces.items():
            print("Generate for", map_size, variants, "...")
            for num_pieces in variants:
                for target_piece in target_pieces:
                    for _ in range(num_tasks):
                        task = Task.create_with_uniform_distractors(map_size, num_pieces, target_piece,
                                                                    colors=colors, shapes=shapes, verbose=False)
                        tasks[split_name].append(task)
        print("Total", len(tasks[split_name]))
        count = 0
        for t in tasks[split_name]:
            if t.target_piece_symbol not in t.piece_symbols:
                count += 1
        print("Problems:", count)
        print("-" * 20)
    store_tasks_to_json(tasks)


if __name__ == '__main__':
    main()
