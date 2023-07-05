import json
from typing import List, Union, Dict

import numpy as np
import random
from cogrip.pentomino.objects import Piece
from cogrip.pentomino.state import Board
from cogrip.pentomino.symbolic.types import SymbolicPiece, RelPositions, Shapes, Rotations
from cogrip.core.grid import GridConfig

from cogrip.constants import COLORS, POSITIONS, SHAPES


def store_tasks_to_json(task_splits, file_name="tasks.json"):
    storable_splits = dict()
    for split_name, tasks in task_splits.items():
        storable_splits[split_name] = [t.to_json() for t in tasks]

    with open(file_name, "w") as f:
        json.dump(storable_splits, f)
    return file_name


def load_tasks_from_json(file_name="tasks.json"):
    if not file_name.endswith(".json"):
        file_name += ".json"
    with open(file_name) as f:
        data = json.load(f)
    task_splits = dict()
    for split_name, tasks in data.items():
        task_splits[split_name] = [Task.from_json(t) for t in tasks]
    return task_splits


class Task:

    def __init__(self, grid_config: GridConfig, pieces: List[Piece], target_piece: Piece, max_steps: int = 100):
        self.idx = None
        self.grid_config = grid_config
        self.pieces = pieces
        self.piece_symbols = [p.piece_config for p in pieces]
        self.target_piece: Piece = target_piece
        self.target_piece_symbol = target_piece.piece_config
        self.max_steps = max_steps

    def create_board(self):
        board = Board(self.grid_config)
        for piece in self.pieces:
            board.add_piece(piece, max_attempts=1, verbose=True)  # check_position should never raise an Exception
        return board

    def to_json(self):
        return {
            "grid_config": self.grid_config.to_dict(),
            "max_steps": self.max_steps,
            "target_piece": self.target_piece.to_json(),
            "pieces": [p.to_json() for p in self.pieces]
        }

    @classmethod
    def from_json(cls, data):
        grid_config = GridConfig.from_dict(data["grid_config"])
        pieces = [Piece.from_json(p) for p in data["pieces"]]
        target_piece = Piece.from_json(data["target_piece"])
        return cls(grid_config, pieces=pieces, target_piece=target_piece, max_steps=data["max_steps"])

    @classmethod
    def create_with_uniform_distractors(cls, map_size: int, num_pieces: int, target_piece_symbol: SymbolicPiece,
                                        verbose: bool = False, colors=None, shapes=None, max_attempts=100):
        grid_config = GridConfig(map_size, map_size, move_step=1, prevent_overlap=True)
        board = Board(grid_config)
        # add target piece
        is_added = board.add_piece_from_symbol(target_piece_symbol, verbose=verbose)
        assert is_added, f"Could not add target piece {target_piece_symbol} to board"
        # add other pieces
        positions = list(POSITIONS)
        positions.remove(RelPositions.CENTER)
        distractor_colors = colors
        if colors is None:
            distractor_colors = list(COLORS)
        distractor_shapes = shapes
        if shapes is None:
            distractor_shapes = list(SHAPES)
        for _ in range(num_pieces - 1):
            for attempt in range(max_attempts):  # if distractor piece symbol cannot be added, then try a different one
                piece_symbol = SymbolicPiece.from_random(distractor_colors, distractor_shapes, positions)
                is_added = board.add_piece_from_symbol(piece_symbol, verbose=verbose)
                if is_added:
                    break
            assert is_added, f"Could not add piece symbol {piece_symbol} to board"
        return cls(grid_config, board.objects, board.objects[0])

    @classmethod
    def create_with_uniform_distractors_from_symbols(cls, map_size: int, num_pieces: int,
                                                     target_piece_symbol: SymbolicPiece,
                                                     distractor_symbols: List[SymbolicPiece],
                                                     verbose: bool = False, max_attempts=100):
        grid_config = GridConfig(map_size, map_size, move_step=1, prevent_overlap=True)
        board = Board(grid_config)
        # add target piece
        is_added = board.add_piece_from_symbol(target_piece_symbol, verbose=verbose)
        assert is_added, f"Could not add target piece {target_piece_symbol} to board"
        # add other pieces
        distractor_symbols = list(distractor_symbols)
        distractor_symbols.remove(target_piece_symbol)  # we do not want to put the same piece twice (at the same pos)
        for _ in range(num_pieces - 1):
            for attempt in range(max_attempts):  # if distractor piece symbol cannot be added, then try a different one
                piece_symbol = random.choice(distractor_symbols)
                is_added = board.add_piece_from_symbol(piece_symbol, verbose=verbose)
                if is_added:
                    break
            assert is_added, f"Could not add piece symbol {piece_symbol} to board"
        return cls(grid_config, board.objects, board.objects[0])

    @classmethod
    def create_random(cls, map_size: int, num_pieces=4,
                      colors=None, shapes=None, positions=None,
                      num_colors: int = None, num_shapes: int = None,
                      orientations: Dict[Shapes, Rotations] = None,
                      verbose: bool = False):
        grid_config = GridConfig(map_size, map_size, move_step=1, prevent_overlap=True)
        board = Board(grid_config)
        if colors is None:
            colors = list(COLORS)
            if num_colors is not None:
                colors = colors[:num_colors]
        if shapes is None:
            shapes = list(SHAPES)
            if num_shapes is not None:
                shapes = shapes[:num_shapes]
        if positions is None:
            positions = list(POSITIONS)
        if RelPositions.CENTER in positions:
            positions.remove(RelPositions.CENTER)
        for _ in range(num_pieces):
            piece_symbol = SymbolicPiece.from_random(colors, shapes, positions)
            if orientations:
                piece_symbol.rotation = orientations[piece_symbol.shape]
            board.add_piece_from_symbol(piece_symbol, verbose=verbose)
        target_piece = np.random.choice(board.objects)
        return cls(grid_config, board.objects, target_piece)


class TaskLoader:

    def __init__(self, tasks, do_shuffle=False):
        assert len(tasks) > 0, "Task list cannot be empty"
        self.tasks = tasks
        self.do_shuffle = do_shuffle
        self.queue = list()

    def next_task(self) -> Task:
        try:
            return self.queue.pop()
        except IndexError:
            self.reset()
            return self.next_task()

    def __len__(self):
        return len(self.tasks)

    def clone(self):
        return TaskLoader(self.tasks, do_shuffle=self.do_shuffle)

    def reset(self):
        self.queue = list(self.tasks)
        if self.do_shuffle:
            random.shuffle(self.queue)

    @classmethod
    def create_taskloader(cls, split_name, task_splits, filter_map_size,
                          filter_num_pieces: Union[int, List], do_shuffle,
                          force_shuffle=False, task_index: int = None, verbose=False):
        task_splits = task_splits[split_name]
        if verbose:
            print("Loaded", len(task_splits), "for", split_name)
        if filter_map_size:
            task_splits = [t for t in task_splits if t.grid_config.width == filter_map_size]
            if verbose:
                print("Filter: map_size == ", filter_map_size)
                print("Remaining", len(task_splits), "for", split_name)
        if filter_num_pieces:
            if isinstance(filter_num_pieces, int):
                filter_num_pieces = [filter_num_pieces]
            task_splits = [t for t in task_splits if len(t.pieces) in filter_num_pieces]
            if verbose:
                print("Filter: num_pieces == ", filter_num_pieces)
                print("Remaining", len(task_splits), "for", split_name)
        if task_index:
            if verbose:
                print("Load only task from remaining at index:", task_index)
            task_splits = [task_splits[task_index]]
        if force_shuffle:
            do_shuffle = True
        elif split_name in ["test", "val", "holdout"]:  # no need to shuffle these
            print(f"Prevent shuffling for {split_name}...")
            do_shuffle = False
        return cls(task_splits, do_shuffle)

    @classmethod
    def from_file(cls, split_name, file_name="tasks.json", do_shuffle=False, force_shuffle=False,
                  filter_map_size=None, filter_num_pieces: Union[int, List] = None,
                  task_index: int = None, verbose=False):
        task_splits = load_tasks_from_json(file_name)
        return cls.create_taskloader(split_name, task_splits, filter_map_size, filter_num_pieces, do_shuffle,
                                     force_shuffle=force_shuffle, task_index=task_index,
                                     verbose=verbose)

    @classmethod
    def all_from_file(cls, file_name="tasks.json", do_shuffle=False,
                      filter_map_size=None, filter_num_pieces: int = None, verbose=False):
        if verbose:
            print("Load all tasks")
            print("--------------")
        task_splits = load_tasks_from_json(file_name)
        loaders = dict([(split_name, cls.create_taskloader(split_name, task_splits,
                                                           filter_map_size, filter_num_pieces, do_shuffle,
                                                           verbose=verbose))
                        for split_name in task_splits])
        if verbose:
            print("-" * 30)
        return loaders, [split_name for split_name in task_splits]
