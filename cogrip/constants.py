from cogrip.pentomino.symbolic.types import Colors, Shapes, RelPositions, Rotations

COLORS = list(Colors)
SHAPES = list(Shapes)
POSITIONS = list(RelPositions)
ROTATIONS = list(Rotations)

COLORS_6 = [Colors.RED, Colors.GREEN, Colors.BLUE, Colors.YELLOW, Colors.BROWN, Colors.PURPLE]
SHAPES_9 = [Shapes.P, Shapes.X, Shapes.T, Shapes.Z, Shapes.W, Shapes.U, Shapes.N, Shapes.F, Shapes.Y]
POSITIONS_8 = [RelPositions.TOP_LEFT, RelPositions.TOP_CENTER, RelPositions.TOP_RIGHT,
               RelPositions.RIGHT_CENTER, RelPositions.BOTTOM_RIGHT, RelPositions.BOTTOM_CENTER,
               RelPositions.BOTTOM_LEFT, RelPositions.LEFT_CENTER]  # no center
ORIENTATIONS = {
    Shapes.P: Rotations.DEGREE_180,
    Shapes.X: Rotations.DEGREE_0,
    Shapes.T: Rotations.DEGREE_270,
    Shapes.Z: Rotations.DEGREE_90,
    Shapes.W: Rotations.DEGREE_0,
    Shapes.U: Rotations.DEGREE_270,
    Shapes.N: Rotations.DEGREE_0,
    Shapes.F: Rotations.DEGREE_0,
    Shapes.Y: Rotations.DEGREE_270
}

# use 0 for "out-of-world object" positions
# use 1 for "no object" positions
COLOR_NAME_TO_IDX = dict((cn, idx) for cn, idx in zip([c.value_name for c in COLORS], range(2, len(COLORS) + 2)))
SHAPE_NAME_TO_IDX = dict((sn, idx) for sn, idx in zip([s.value for s in SHAPES], range(2, len(SHAPES) + 2)))

IDX_TO_COLOR_NAME = dict((idx, cn) for cn, idx in zip([c.value_name for c in COLORS], range(2, len(COLORS) + 2)))
IDX_TO_COLOR_NAME[0] = "oow"
IDX_TO_COLOR_NAME[1] = "empty"

IDX_TO_SHAPE_NAME = dict((idx, sn) for sn, idx in zip([s.value for s in SHAPES], range(2, len(SHAPES) + 2)))
IDX_TO_SHAPE_NAME[0] = "oow"
IDX_TO_SHAPE_NAME[1] = "empty"
