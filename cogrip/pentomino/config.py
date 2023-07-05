from cogrip.pentomino.objects import ShapesMatrix
from cogrip.pentomino.symbolic.types import Colors
from cogrip.core.config import Config


def load_shapes_as_dict():
    return dict([(shape.name, shape.value) for shape in list(ShapesMatrix)])


def load_colors_as_list():
    return [c.value_hex for c in list(Colors)]


class PentoConfig(Config):
    """ Alternative to a config.json file """

    def __init__(self, width: int = 40, height: int = 40):
        # not all parameters configurable yet
        super(PentoConfig, self).__init__(type_config=load_shapes_as_dict(),
                                          width=width,
                                          height=height,
                                          snap_to_grid=True,
                                          prevent_overlap=True,
                                          move_step=1,
                                          actions=["move", "rotate", "flip", "grip"],
                                          colors=load_colors_as_list()
                                          )
