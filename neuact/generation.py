import itertools
import random
from collections import defaultdict
from typing import List, Dict, Set

from cogrip.pentomino.symbolic.types import SymbolicPiece, Shapes, Colors, RelPositions, Rotations


def check_pos_split_counts(configs_by_split: Dict):
    for k, v in configs_by_split.items():
        print("Split", k, "count(s,c,p):", len(v))


def check_piece_pos_per_split(configs_by_pos_split: Dict):
    for split_name, pieces in configs_by_pos_split.items():
        print("Split:", split_name)
        __check_piece_pos(pieces)
        print()


def __check_piece_pos(pieces: Set[SymbolicPiece]):
    n_pieces = defaultdict(set)
    for p in pieces:
        n_pieces[(p.shape, p.color)].add(p.rel_position)
    keys = sorted(list(n_pieces.keys()))
    n_pos = set()
    for k in keys:
        v = n_pieces[k]
        print(f"Piece ({len(list(v))}:", k, sorted(list(v)))
        n_pos.update(v)
    print(f"Positions ({len(n_pos)})", n_pos)


def check_color_split_counts(configs_by_split: Dict, num_positions):
    for k, v in configs_by_split.items():
        print("Split", k, "count(s,c,p):", len(v), "count(s,c):", int(len(v) / num_positions))


def check_piece_colors_per_split(configs_by_color_split: Dict):
    for split_name, pieces in configs_by_color_split.items():
        print("Split:", split_name)
        __check_piece_colors(pieces)
        print()


def __check_piece_colors(pieces: Set):  # ignore position
    n_pieces = defaultdict(set)
    for p in pieces:
        n_pieces[p.shape].add(p.color)
    keys = sorted(list(n_pieces.keys()))
    n_colors = set()
    for k in keys:
        v = n_pieces[k]
        print(f"Piece ({len(v)}):", k, sorted(list(v)))
        n_colors.update(v)
    print(f"Colors ({len(n_colors)})", n_colors)


def __create_rotated_color_splits(shape_idx, num_colors, num_positions, color_holdout=1):
    splits = ["train"] * (num_colors * num_positions)
    pointer = shape_idx * num_positions  # for each shape, start "one whole" color later
    for _ in range(color_holdout * num_positions):  # hold out "the whole" color (all positions)
        splits[pointer % (num_colors * num_positions)] = "test"
        pointer += 1
    return splits


def create_color_holdout(shapes: List[Shapes], colors: List[Colors], positions: List[RelPositions],
                         color_holdout=1, verbose=False):
    configs_by_split = {
        "train": set(),
        "test": set()
    }
    for shape_idx, shape in enumerate(shapes):
        splits = __create_rotated_color_splits(shape_idx, num_colors=len(colors), num_positions=len(positions),
                                               color_holdout=color_holdout)
        if verbose:
            print(splits)
        for split, (color, position) in zip(splits, itertools.product(colors, positions)):
            rotation = random.choice(list(Rotations))  # lets assume a random rotation
            piece_config = SymbolicPiece(color=color, shape=shape, rel_position=position, rotation=rotation)
            configs_by_split[split].add(piece_config)
    return configs_by_split
