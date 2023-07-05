import random
from collections import defaultdict
from typing import List, Dict, Set

from cogrip.pentomino.symbolic.types import SymbolicPiece, Colors, Shapes, RelPositions, PropertyNames, SymbolicPieceGroup, \
    Rotations


class RestrictivePieceConfigGroupSampler:

    def __init__(self, pieces: List[SymbolicPiece], pieces_per_pos: int = 2):
        """

        :param pieces: without the target_piece
        """
        self.pieces_per_pos = pieces_per_pos
        self.pieces_by_color: Dict[Colors, List[SymbolicPiece]] = SymbolicPiece.group_by_color(pieces)
        self.pieces_by_shape: Dict[Shapes, List[SymbolicPiece]] = SymbolicPiece.group_by_shape(pieces)
        self.pieces_by_pos: Dict[RelPositions, List[SymbolicPiece]] = SymbolicPiece.group_by_pos(pieces)
        self.meta = {
            PropertyNames.COLOR: self.pieces_by_color,
            PropertyNames.SHAPE: self.pieces_by_shape,
            PropertyNames.REL_POSITION: self.pieces_by_pos,
        }

    def __check_allowed_positions(self, allowed_positions, n_pieces):
        num_pos = len(allowed_positions)
        num_possible = self.pieces_per_pos * num_pos
        if num_possible < n_pieces:
            raise Exception(f"with pieces_per_pos={self.pieces_per_pos} and num_pos={num_pos} "
                            f"there can be maximal n_pieces={num_possible} in the set")

    def __check_and_get_allowed_positions(self, n_pieces):
        allowed_positions = list(self.pieces_by_pos.keys())  # all positions, where there are pieces for
        self.__check_allowed_positions(allowed_positions, n_pieces)
        return allowed_positions

    def __check_and_reduce(self, allowed_positions, pos_counts):
        for pos, counts in pos_counts.items():
            if counts >= self.pieces_per_pos:
                if pos in allowed_positions:
                    allowed_positions.remove(pos)

    def sample_special(self, target_piece: SymbolicPiece, n_pieces: int):
        """
        color,shape,position:
                    1 x Share(color), 1 Share(shape), Diff(pos)
                    1 x Share(color), 1 Diff(shape), Diff(pos),
                 others Any(color), Any(shape), Any(pos)
        """
        # hierarchical sampling:
        # 1. sample position (so we can block certain positions if drawn already twice)
        # 2. sample piece on position
        if n_pieces < 3:
            print(f"Warn: (color,shape,position) requires at least 3 distractors, but only {n_pieces} given. "
                  "Otherwise it will reduce to (shape,position).")

        allowed_positions = self.__check_and_get_allowed_positions(n_pieces)
        pos_counts = dict([(pos, 0) for pos in allowed_positions])

        piece_set = []

        # one dist must share color and shape, otherwise IA stops already
        possible_pieces = self.pieces_by_color[target_piece.color]
        possible_pieces = [p for p in possible_pieces if p.shape == target_piece.shape]
        possible_pieces = [p for p in possible_pieces if p.rel_position != target_piece.rel_position]
        piece = random.choice(possible_pieces)
        pos_counts[piece.rel_position] += 1
        piece_set.append(piece)

        # one dist must share color, but not shape, so that shape must be mentioned
        possible_pieces = self.pieces_by_color[target_piece.color]
        possible_pieces = [p for p in possible_pieces if p.shape != target_piece.shape]
        possible_pieces = [p for p in possible_pieces if p.rel_position != target_piece.rel_position]
        piece = random.choice(possible_pieces)
        pos_counts[piece.rel_position] += 1
        piece_set.append(piece)

        if n_pieces <= 2:  # we are already finished (this is only (shape,position) though)
            return SymbolicPieceGroup(piece_set)

        # remove already, if we have reached the limit at a pos
        self.__check_and_reduce(allowed_positions, pos_counts)

        # other piece have any shape and position, but not color, so that color must be mentioned
        # Note: target piece is not in pieces already
        for _ in range(n_pieces - 2):
            pos = random.choice(allowed_positions)
            possible_pieces = self.pieces_by_pos[pos]
            possible_pieces = [p for p in possible_pieces if p.color != target_piece.color]
            piece = random.choice(possible_pieces)
            piece_set.append(piece)
            pos_counts[piece.rel_position] += 1
            self.__check_and_reduce(allowed_positions, pos_counts)
        return SymbolicPieceGroup(piece_set)

    def sample_some_with_prop1_and_position(self, target_piece: SymbolicPiece, prop1: PropertyNames, n_pieces: int):
        # hierarchical sampling:
        # 1. sample position (so we can block certain positions if drawn already twice)
        # 2. sample piece on position
        # Note: if you dont need this, simply use random.sample(n,k)
        piece_set = []

        # exactly on with the same position, but different prop
        possible_pieces = self.pieces_by_pos[target_piece.rel_position]
        possible_pieces = [p for p in possible_pieces if p[prop1] != target_piece[prop1]]
        piece1 = random.choice(possible_pieces)
        piece_set.append(piece1)

        # others with same prop, but different position
        possible_pieces = self.meta[prop1][target_piece[prop1]]
        possible_pieces = [p for p in possible_pieces if p.rel_position != target_piece.rel_position]
        possible_pieces_by_pos = SymbolicPiece.group_by_pos(possible_pieces)

        # positions are defined by the other possible pieces that do not share the target piece position
        allowed_positions = [p.rel_position for p in possible_pieces]
        pos_counts = dict([(pos, 0) for pos in allowed_positions])

        for _ in range(n_pieces - 1):
            pos = random.choice(allowed_positions)
            piece = random.choice(possible_pieces_by_pos[pos])
            piece_set.append(piece)
            pos_counts[piece.rel_position] += 1
            self.__check_and_reduce(allowed_positions, pos_counts)
        return SymbolicPieceGroup(piece_set)

    def __get_same_but_diff(self, target_piece: SymbolicPiece,
                            same_prop: PropertyNames, diff_prop: PropertyNames, pos):
        possible_pieces = self.meta[same_prop][target_piece[same_prop]]
        possible_pieces = [p for p in possible_pieces if p[diff_prop] != target_piece[diff_prop]]
        possible_pieces = [p for p in possible_pieces if p.rel_position == pos]
        if not possible_pieces:
            print("target_piece:", target_piece)
            print("same_prop:", same_prop)
            print("diff_prop:", diff_prop)
            print("pos:", pos)
            print(self.meta[same_prop][target_piece[same_prop]])
        return possible_pieces

    def sample_some_with_prop1_and_prop2(self, target_piece: SymbolicPiece, prop1: PropertyNames, prop2: PropertyNames,
                                         n_pieces: int):
        # hierarchical sampling:
        # 1. sample position (so we can block certain positions if drawn already twice)
        # 2. sample piece on position
        # Note: if you dont need this, simply use random.sample(n,k)
        allowed_positions = self.__check_and_get_allowed_positions(n_pieces)
        pos_counts = dict([(pos, 0) for pos in allowed_positions])
        piece_set = []
        # select at least one, but never all for the first property
        split_size = random.randint(1, n_pieces - 1)
        for _ in range(split_size):
            pos = random.choice(allowed_positions)
            possible_pieces = self.__get_same_but_diff(target_piece, prop1, prop2, pos)
            piece = random.choice(possible_pieces)
            piece_set.append(piece)
            pos_counts[piece.rel_position] += 1
            self.__check_and_reduce(allowed_positions, pos_counts)
        # select the remaining ones for the second property
        for _ in range(n_pieces - split_size):
            pos = random.choice(allowed_positions)
            possible_pieces = self.__get_same_but_diff(target_piece, prop2, prop1, pos)
            piece = random.choice(possible_pieces)
            piece_set.append(piece)
            pos_counts[piece.rel_position] += 1
            self.__check_and_reduce(allowed_positions, pos_counts)
        return SymbolicPieceGroup(piece_set)

    def sample_with_position_restriction(self, n_pieces: int, disallow_pos: List[RelPositions] = None):
        if disallow_pos is None:
            disallow_pos = list()
        # hierarchical sampling:
        # 1. sample position (so we can block certain positions if drawn already twice)
        # 2. sample piece on position
        # Note: if you dont need this, simply use random.sample(n,k)
        allowed_positions = self.__check_and_get_allowed_positions(n_pieces)
        # if we allow all positions, but there are only pieces of the same shape and color
        # then there will be no pieces available at the target piece position e.g. for position-only utterances
        for pos in disallow_pos:
            if pos in allowed_positions:
                allowed_positions.remove(pos)
        pos_counts = dict([(pos, 0) for pos in allowed_positions])
        piece_set = []
        for _ in range(n_pieces):
            pos = random.choice(allowed_positions)
            possible_pieces = self.pieces_by_pos[pos]
            piece = random.choice(possible_pieces)
            piece_set.append(piece)
            pos_counts[piece.rel_position] += 1
            self.__check_and_reduce(allowed_positions, pos_counts)
        return SymbolicPieceGroup(piece_set)


class UtteranceTypeOrientedDistractorSetSampler:

    def __init__(self, pieces: List[SymbolicPiece], target_piece: SymbolicPiece, n_retries=100):
        self.n_retries = n_retries
        # remove the target from the piece set
        self.pieces = list(pieces)
        self.pieces.remove(target_piece)
        # group pieces by their property values
        pieces_by_value = defaultdict(list)
        for piece in self.pieces:
            for pn in list(PropertyNames):
                pieces_by_value[piece[pn]].append(piece)
        self.pieces_by_value = pieces_by_value
        self.target_piece = target_piece

    def sample_many_distractor_groups(self, utterance_type, n_sets, pieces_per_set, verbose=False, rotate_pieces=False):
        n_min, n_max = pieces_per_set[0], pieces_per_set[1]
        distractors_sets = set()
        retries = 0
        while len(distractors_sets) < n_sets and retries < self.n_retries:
            set_size = random.randint(n_min, n_max) - 1  # already 1 reserved for target piece
            distractor_set = self.create_distractor_configs_csp(unique_props=set(utterance_type),
                                                                num_distractors=set_size)
            if rotate_pieces:
                for d in distractor_set:
                    d.rotation = Rotations.from_random()
            if distractor_set in distractors_sets:  # avoid duplicates (if sample space is small)
                retries += 1
                continue
            distractors_sets.add(distractor_set)
        if verbose and len(distractors_sets) < n_sets:
            print("Warn: Too many duplicates. Sampled sets:", len(distractors_sets))
        if verbose and retries > 0:
            print("Warn: Retries", retries)
        return distractors_sets

    def create_distractor_configs_csp(self, unique_props: Set[PropertyNames], num_distractors: int = 1):
        """
        Note: The preference order is fix [PropertyNames.COLOR, PropertyNames.SHAPE, PropertyNames.REL_POSITION]

        So, if we want to generate distractors for

            color:    any other distractor, but different colors
                        Diff(color), Any(shape), Any(pos)
            shape:    all having the same color (no exclusions), but different shapes; random positions
                        All(color), Diff(shape), Any(pos)
            position: all having the same color and shape (no exclusions), but different positions
                        All(color), All(shape), Diff(pos)

            color,shape:    at least one (and never all) distractor shares the color or the shape
                        Some(color), Some(shape), Any(pos)
            color,position: at least one (and never all) distractor shares the color or the position; all same shapes
                        Some(color), All(shape), Some(pos)
            shape,position: at least one (and never all) distractor shares the shape or the position; all same color
                        All(color), Some(shape), Some(pos)

            color,shape,position:
                        Some(color), Some(shape), Some(pos)

        :return:
        """
        if num_distractors < 1:
            raise Exception(f"There must be at least one distractor, but num_distractors is {num_distractors}")

        # When we have a single uniq prop, then all other pieces are disallowed to have that prop
        if len(unique_props) == 1:
            unique_prop = list(unique_props)[0]
            disallowed_positions = []
            """
            color:    any other distractor, but different colors
                        Diff(color), Any(shape), Any(pos)
            """
            if unique_prop == PropertyNames.COLOR:
                # all pieces which do not share the same color
                possible_distractors = [piece for piece in self.pieces if piece.color != self.target_piece.color]

            """
            shape:    all having the same color (no exclusions), but different shapes; random positions
                        All(color), Diff(shape), Any(pos)
            """
            if unique_prop == PropertyNames.SHAPE:
                possible_distractors = self.pieces_by_value[self.target_piece.color]
                possible_distractors = [piece for piece in possible_distractors
                                        if piece.shape != self.target_piece.shape]

            """
            position: all having the same color and shape (no exclusions), but different positions
                        All(color), All(shape), Diff(pos)
            """
            if unique_prop == PropertyNames.REL_POSITION:
                possible_distractors = self.pieces_by_value[self.target_piece.color]
                possible_distractors = [piece for piece in possible_distractors
                                        if piece.shape == self.target_piece.shape]
                possible_distractors = [piece for piece in possible_distractors
                                        if piece.rel_position != self.target_piece.rel_position]
                disallowed_positions.append(self.target_piece.rel_position)

            sampler = RestrictivePieceConfigGroupSampler(possible_distractors)
            return sampler.sample_with_position_restriction(num_distractors, disallow_pos=disallowed_positions)

        if len(unique_props) == 2:
            """
            color,shape:    at least one (and never all) distractor shares the color or the shape
                        Some(color), Some(shape), Any(pos)
            """
            if PropertyNames.COLOR in unique_props and PropertyNames.SHAPE in unique_props:
                sampler = RestrictivePieceConfigGroupSampler(self.pieces)
                return sampler.sample_some_with_prop1_and_prop2(self.target_piece,
                                                                PropertyNames.COLOR,
                                                                PropertyNames.SHAPE,
                                                                n_pieces=num_distractors)
            """
            color,position: at least one (and never all) distractor shares the color or the position; all same shapes
                        Some(color), All(shape), Some(pos)
            """
            if PropertyNames.COLOR in unique_props and PropertyNames.REL_POSITION in unique_props:
                possible_distractors = self.pieces_by_value[self.target_piece.shape]
                sampler = RestrictivePieceConfigGroupSampler(possible_distractors)
                return sampler.sample_some_with_prop1_and_position(self.target_piece,
                                                                   PropertyNames.COLOR,
                                                                   n_pieces=num_distractors)
            """
            shape,position: at least one (and never all) distractor shares the shape or the position; all same color
                        All(color), Some(shape), Some(pos)
            """
            if PropertyNames.SHAPE in unique_props and PropertyNames.REL_POSITION in unique_props:
                possible_distractors = self.pieces_by_value[self.target_piece.color]
                sampler = RestrictivePieceConfigGroupSampler(possible_distractors)
                return sampler.sample_some_with_prop1_and_position(self.target_piece,
                                                                   PropertyNames.SHAPE,
                                                                   n_pieces=num_distractors)

        if len(unique_props) == 3:
            """
            color,shape,position:
                        1 x Share(color), 1 Share(shape), Diff(pos)
                        1 x Share(color), 1 Diff(shape), Diff(pos), 
                     others Any(color), Any(shape), Any(pos)
            """
            sampler = RestrictivePieceConfigGroupSampler(self.pieces)
            return sampler.sample_special(self.target_piece, n_pieces=num_distractors)
