from collections import defaultdict
from enum import Enum, IntEnum
import random
from typing import List, Tuple, Dict


class Shapes(Enum):
    F = "F"
    I = "I"
    L = "L"
    N = "N"
    P = "P"
    T = "T"
    U = "U"
    V = "V"
    W = "W"
    X = "X"
    Y = "Y"
    Z = "Z"

    def __repr__(self):
        return f"{self.value}"

    def __key(self):
        return self.value

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def __lt__(self, other):
        return self.value.__lt__(other.value)

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, value):
        return cls[value]


class Rotations(IntEnum):
    DEGREE_0 = 0
    DEGREE_90 = 90
    DEGREE_180 = 180
    DEGREE_270 = 270

    def __repr__(self):
        return f"{self.value}"

    def __key(self):
        return self.value

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def __lt__(self, other):
        return self.value.__lt__(other.value)

    def to_json(self):
        return self.value

    @classmethod
    def from_random(cls):
        possible_rotations = list(cls)
        return random.choice(possible_rotations)

    @classmethod
    def from_json(cls, value):
        return cls[f"DEGREE_{value}"]


class Colors(Enum):
    RED = ("red", "#ff0000", [255, 0, 0])
    ORANGE = ("orange", "#ffa500", [255, 165, 0])
    YELLOW = ("yellow", "#ffff00", [255, 255, 0])
    GREEN = ("green", "#008000", [0, 128, 0])
    BLUE = ("blue", "#0000ff", [0, 0, 255])
    CYAN = ("cyan", "#00ffff", [0, 255, 255])
    PURPLE = ("purple", "#800080", [128, 0, 128])
    BROWN = ("brown", "#8b4513", [139, 69, 19])
    GREY = ("grey", "#808080", [128, 128, 128])
    PINK = ("pink", "#ffc0cb", [255, 192, 203])
    OLIVE_GREEN = ("olive green", "#808000", [128, 128, 0])  # dark yellowish-green
    NAVY_BLUE = ("navy blue", "#000080", [0, 0, 128])  # dark blue

    def __init__(self, value_name, value_hex, value_rgb):
        self.value_name = value_name
        self.value_hex = value_hex
        self.value_rgb = value_rgb

    def __repr__(self):
        return f"{self.value_name}"

    def __key(self):
        return self.value_name

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return self.value_name

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value_name == other.value_name
        return False

    def __lt__(self, other):
        return self.value_name.__lt__(other.value_name)

    def to_tuple(self):
        return tuple(self.value)

    @classmethod
    def from_tuple(cls, t):
        return cls.from_json(t[0])

    def to_json(self):
        return self.value_name

    @classmethod
    def from_json(cls, value_name):
        return cls[value_name.upper().replace(" ", "_")]


class RelPositions(Enum):
    TOP_LEFT = "top left"
    TOP_CENTER = "top center"
    TOP_RIGHT = "top right"
    RIGHT_CENTER = "right center"
    BOTTOM_RIGHT = "bottom right"
    BOTTOM_CENTER = "bottom center"
    BOTTOM_LEFT = "bottom left"
    LEFT_CENTER = "left center"
    CENTER = "center"

    def __repr__(self):
        return f"{self.value}"

    def __key(self):
        return self.value

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return self.value

    def __lt__(self, other):
        return self.value.__lt__(other.value)

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, value):
        return cls[value.upper().replace(" ", "_")]

    def to_random_coords(self, board_width, board_height):
        # the relative positions are derived from their own "grid"-like board
        # with 3,3 there are as many RelPositions as cells in the grid, but
        # we could have also "thinner" slices or put more "space" onto the edges
        # TODO there is no general case yet
        num_cols, num_rows = 3, 3
        width_step, height_step = board_width // num_cols, board_height // num_rows

        # Uff, this is a bit harsh.
        # Pieces coords is their upper-left corner!
        # They are furthermore drawn on potentially 5x5 grids.
        piece_grid_size = 2

        # So when we sample (0,0) then the piece is in the upper left corner fully fit,
        # but when we sample something at the right or bottom, then pieces cannot be fully drawn anymore
        # so the actually possible coordinate space is smaller than what is shown on the board.
        # We apply the "padding" for all max values at the end of this method.
        x_min, x_max = 0, board_width
        y_min, y_max = 0, board_height

        # This is in particular difficult, because we "see" the pieces on other coords (e.g. the center of a piece).
        # So given the coords, an algorithm must actually "imagine" where the piece is actually drawn using an offset
        # and cannot simply derive this from the coords itself
        x_left = 0, width_step
        x_right = 2 * width_step, board_width

        y_top = 0, height_step
        y_bottom = 2 * height_step, board_height

        x_center = width_step, 2 * width_step
        y_center = height_step, 2 * height_step

        if self == RelPositions.TOP_LEFT:
            x_min, x_max = x_left
            y_min, y_max = y_top
        if self == RelPositions.TOP_CENTER:
            x_min, x_max = x_center
            y_min, y_max = y_top
        if self == RelPositions.TOP_RIGHT:
            x_min, x_max = x_right
            y_min, y_max = y_top
        if self == RelPositions.RIGHT_CENTER:
            x_min, x_max = x_right
            y_min, y_max = y_center
        if self == RelPositions.BOTTOM_RIGHT:
            x_min, x_max = x_right
            y_min, y_max = y_bottom
        if self == RelPositions.BOTTOM_CENTER:
            x_min, x_max = x_center
            y_min, y_max = y_bottom
        if self == RelPositions.BOTTOM_LEFT:
            x_min, x_max = x_left
            y_min, y_max = y_bottom
        if self == RelPositions.LEFT_CENTER:
            x_min, x_max = x_left
            y_min, y_max = y_center
        if self == RelPositions.CENTER:
            x_min, x_max = x_center
            y_min, y_max = y_center
        return random.randint(x_min, x_max - piece_grid_size), random.randint(y_min, y_max - piece_grid_size)

    @staticmethod
    def from_coords(x, y, board_width, board_height):
        # the relative positions are derived from their own "grid"-like board
        # with 3,3 there are as many RelPositions as cells in the grid, but
        # we could have also "thinner" slices or put more "space" onto the edges
        # TODO there is no general case yet
        num_cols, num_rows = 3, 3
        width_step, height_step = board_width // num_cols, board_height // num_rows
        # x = obj.x + (obj.width / 2)
        # y = obj.y + (obj.height / 2)
        # TODO this actually looks wrong
        pos = None
        if y < 1 * height_step:
            pos = RelPositions.TOP_CENTER
        elif y >= 2 * height_step:
            pos = RelPositions.BOTTOM_CENTER
        if x < 1 * width_step:
            if pos == RelPositions.TOP_CENTER:
                return RelPositions.TOP_LEFT
            if pos == RelPositions.BOTTOM_CENTER:
                return RelPositions.BOTTOM_LEFT
            return RelPositions.LEFT_CENTER
        elif x >= 2 * width_step:
            if pos == RelPositions.TOP_CENTER:
                return RelPositions.TOP_RIGHT
            if pos == RelPositions.BOTTOM_CENTER:
                return RelPositions.BOTTOM_RIGHT
            return RelPositions.RIGHT_CENTER
        if pos:
            return pos
        return RelPositions.CENTER


class PropertyNames(Enum):
    COLOR = "color"
    SHAPE = "shape"
    REL_POSITION = "rel_position"
    ROTATION = "rotation"

    def __repr__(self):
        return f"{self.value}"

    def __key(self):
        return self.value

    def __hash__(self):
        return hash(self.__key())

    def __str__(self):
        return self.value

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.value == other.value
        return False

    def __lt__(self, other):
        return self.value.__lt__(other.value)

    def to_json(self):
        return self.value

    @classmethod
    def from_json(cls, value):
        return PropertyNames[value.upper()]

    @classmethod
    def from_string(cls, name):
        for pn in list(cls):
            if pn.value == name:
                return pn
        return None


class SymbolicPiece:
    """ Symbolic piece representation consisting of a tuple of discrete colors, shapes and positions"""

    def __init__(self, color: Colors = None, shape: Shapes = None, rel_position: RelPositions = None,
                 rotation=Rotations.DEGREE_0):
        self.color = color
        self.shape = shape
        self.rel_position = rel_position
        self.rotation = rotation

    def is_undefined(self):
        return self.color is None and self.shape is None and self.rel_position is None

    def __getitem__(self, prop_name: PropertyNames):
        if prop_name == PropertyNames.COLOR:
            return self.color
        if prop_name == PropertyNames.SHAPE:
            return self.shape
        if prop_name == PropertyNames.REL_POSITION:
            return self.rel_position
        if prop_name == PropertyNames.ROTATION:
            return self.rotation
        raise Exception(f"Cannot get {prop_name}")

    def __setitem__(self, prop_name: PropertyNames, value):
        if prop_name == PropertyNames.COLOR:
            self.color = value
            return
        if prop_name == PropertyNames.SHAPE:
            self.shape = value
            return
        if prop_name == PropertyNames.REL_POSITION:
            self.rel_position = value
            return
        if prop_name == PropertyNames.ROTATION:
            self.rotation = value
            return
        raise Exception(f"Cannot set {prop_name}.")

    def __repr__(self):
        return f"({self.shape}, {self.color}, {self.rel_position})"

    def __str__(self):
        return f"({self.shape}, {self.color}, {self.rel_position})"

    def __key(self):
        return self.shape, self.color, self.rel_position

    def __hash__(self):
        return hash(self.__key())

    def __lt__(self, other):
        return self.__key() < other.__key()

    def __eq__(self, other):
        if isinstance(other, SymbolicPiece):
            return self.__key() == other.__key()
        if other is None:
            return False
        raise ValueError(f"Other is not {self.__class__} but {other.__class__}")

    def copy(self):
        return SymbolicPiece(self.color, self.shape, self.rel_position)

    def to_json(self):
        return self.color.to_json(), self.shape.to_json(), self.rel_position.to_json(), self.rotation.to_json()

    @classmethod
    def from_json(cls, t: Tuple):
        return cls(Colors.from_json(t[0]), Shapes.from_json(t[1]), RelPositions.from_json(t[2]),
                   Rotations.from_json(t[3]))

    @classmethod
    def from_dict(cls, d: Dict):
        return cls(Colors.from_json(d["color"]), Shapes.from_json(d["shape"]),
                   RelPositions.from_json(d["rel_position"]),
                   Rotations.from_json(d["rotation"]))

    @classmethod
    def from_random(cls, colors, shapes, rel_positions, rotations=None):
        if rotations is None:
            return cls(random.choice(colors), random.choice(shapes), random.choice(rel_positions))
        return cls(random.choice(colors), random.choice(shapes), random.choice(rel_positions), random.choice(rotations))

    @staticmethod
    def group_by_pos(pieces: List):
        groups = defaultdict(list)
        for piece in pieces:
            groups[piece.rel_position].append(piece)
        # print("group_by_pos", [len(groups[p]) for p in groups])
        return groups

    @staticmethod
    def group_by_color(pieces: List):
        groups = defaultdict(list)
        for piece in pieces:
            groups[piece.color].append(piece)
        # print("group_by_color", [len(groups[p]) for p in groups])
        return groups

    @staticmethod
    def group_by_shape(pieces: List):
        groups = defaultdict(list)
        for piece in pieces:
            groups[piece.shape].append(piece)
        # print("group_by_shape", [len(groups[p]) for p in groups])
        return groups


class SymbolicPieceGroup:
    """ Multiple symbolic pieces represented together as a comparable entity (order does not matter)"""

    def __init__(self, pieces: List[SymbolicPiece]):
        self.pieces = pieces  # allow duplicates and preserve order

    def __getitem__(self, item):
        return self.pieces[item]

    def __repr__(self):
        return f"PCG{self.pieces}"

    def __str__(self):
        return f"({self.pieces})"

    def __iter__(self):
        return self.pieces.__iter__()

    def __len__(self):
        return len(self.pieces)

    def __key(self):
        return tuple(sorted(self.pieces))  # we ignore order for comparison

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        if isinstance(other, SymbolicPieceGroup):
            return self.__key() == other.__key()
        raise ValueError(f"Other is not {self.__class__} but {other.__class__}")

    def to_json(self):
        return [p.to_json() for p in self.pieces]

    @classmethod
    def from_json(cls, pieces: List):
        return cls([SymbolicPiece.from_json(p) for p in pieces])
