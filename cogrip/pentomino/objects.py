from enum import Enum
from cogrip.pentomino.symbolic.types import SymbolicPiece, Colors, Shapes, RelPositions, Rotations
from cogrip.core.grid import GridConfig
from cogrip.core.obj import Obj


class ShapesMatrix(Enum):
    F = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    I = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]
    L = [
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    N = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 1, 0, 0]
    ]
    P = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    T = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    U = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    V = [
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    W = [
        [0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ]
    X = [
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    Y = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [1, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ]
    Z = [
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ]


class Piece(Obj):
    """ A pentomino specific wrapper around GOLMi Obj """

    def __init__(self, piece_config: SymbolicPiece, obj_kwargs):
        """ Todo: consider to loose the relation between piece_obj (realisation) and piece_config (symbol)"""
        super().__init__(**obj_kwargs)
        self.piece_config = piece_config

    def __repr__(self):
        return f"{self.id_n}, {repr(self.piece_config)}"

    def __str__(self):
        return f"{self.id_n}, {repr(self.piece_config)}"

    def to_json(self):
        return {
            "piece_symbol": self.piece_config.to_json() if self.piece_config is not None else None,
            "piece_obj": self.to_dict()
        }

    @classmethod
    def from_json(cls, data):
        piece_symbol = data["piece_symbol"]
        piece_config = SymbolicPiece.from_json(piece_symbol) if piece_symbol is not None else None
        return cls(piece_config, obj_kwargs=data["piece_obj"])

    @classmethod
    def from_dict(cls, obj_dict, grid_config: GridConfig = None):
        piece_obj = Obj.from_dict(obj_dict)  # todo: this seems just like a workaround
        piece_config = None
        if grid_config:  # we can only know the position, when we know the board size
            piece_config = SymbolicPiece(Colors.from_tuple(piece_obj.color),
                                         Shapes.from_json(piece_obj.type),
                                         RelPositions.from_coords(piece_obj.x, piece_obj.y,
                                                                  grid_config.width, grid_config.height),
                                         Rotations.from_json(int(piece_obj.rotation)))
        return cls(piece_config, piece_obj.to_dict())

    @classmethod
    def from_symbol(cls, piece_id, piece_symbol, board_width, board_height):
        x, y = piece_symbol.rel_position.to_random_coords(board_width, board_height)
        shape_matrix = ShapesMatrix[piece_symbol.shape.value]
        obj_dict = dict(id_n=piece_id,
                        type=piece_symbol.shape.value,
                        x=x, y=y,
                        block_matrix=shape_matrix.value,
                        color=piece_symbol.color.to_tuple())  # (name, hex, rgb)
        piece_obj = Piece(piece_symbol, obj_dict)
        if piece_symbol.rotation:
            piece_obj.rotate(piece_symbol.rotation.value)
        return piece_obj
