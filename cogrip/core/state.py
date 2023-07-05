import json

from typing import Dict, List, TypeVar, Generic, Union

from cogrip.core import Jsonable
from cogrip.core.obj import Obj, Objects
from cogrip.core.gripper import Gripper, Grippers
from cogrip.core.grid import Grid, GridConfig

T = TypeVar('T', bound=Obj)


class State(Generic[T], Jsonable):
    def __init__(self, grid_config: GridConfig,
                 objects: Objects = None,
                 grippers: Grippers = None,
                 targets: Objects = None,
                 state_id: int = None,
                 global_id: int = None):
        self.state_id = state_id
        self.global_id = global_id

        self.grippers = grippers if grippers else Grippers()
        self.objects = objects if objects else Objects()
        self.targets = targets if targets else Objects()

        assert grid_config is not None, "grid_config must not be None"
        self.grid_config = grid_config

        self.object_grid = Grid.create_from_config(grid_config)
        for obj in self.objects:
            self.object_grid.add_obj(obj)

        self.target_grid = Grid.create_from_config(grid_config) if targets else None
        if self.target_grid:
            for target in self.targets:
                self.target_grid.add_obj(target)

    def to_json(self):
        return self.to_state_dict(include_grid_config=True)

    def move_gr(self, gr_id, dx, dy):
        """
        Change gripper position by moving in direction (dx, dy).
        @param gr_id 	id of the gripper to move
        @param dx 	x direction
        @param dy 	y direction
        """
        self.grippers[gr_id].x += dx
        self.grippers[gr_id].y += dy

    def move_obj(self, obj: Union[int, Obj], dx, dy):
        """
         Change an object's position by moving in direction (dx, dy).
         @param obj 	object or id
         @param dx 	x direction
         @param dy 	y direction
        """
        if isinstance(obj, int):
            obj = self.objects[obj]
        obj.x += dx
        obj.y += dy

    def rotate_obj(self, obj_id, d_angle):
        """
        Change an object's goal_rotation by d_angle.
        @param obj_id  	object id
        @param d_angle	current angle is changed by d_angle
        """
        if d_angle != 0:
            self.objects[obj_id].rotate(d_angle)

    def flip_obj(self, obj_id):
        """
        Mirror an object.
        @param obj_id 	object_id
        """
        # change 'mirrored' attribute
        self.objects[obj_id].flip()

    def grip(self, gr_id, obj_id):
        """
        Attach a given object to the gripper.
        @param gr_id 	id of the gripper that grips obj_id
        @param obj_id 	id of object to grip, must be in objects
         """
        self.objects[obj_id].gripped = True
        self.grippers[gr_id].gripped_obj = self.objects[obj_id]

    def ungrip(self, gr_id):
        """
        Detach the currently gripped object from the gripper.
        @param gr_id 	id of the gripper that ungrips
        """
        gripper = self.grippers[gr_id]
        gripped_obj = self.objects[gripper.gripped]
        gripped_obj.gripped = False
        gripper.gripped_obj = None

    @classmethod
    def from_json(cls, filename, type_config, config):
        """
        @param filename String, name of a json file describing a State.
                        The key "type_config" mapping to a dict is mandatory.
        @param type_config  dict mapping type names to block matrices
        @return new State instance with the given attributes
        """
        with open(filename, mode="r") as file:
            json_data = json.loads(file.read())
        return cls.from_state_dict(json_data, type_config, config)

    @classmethod
    # TODO: make sure pieces are on the board! (at least emit warning)
    def from_state_dict(cls, source_dict, type_config=None, grid_config: GridConfig = None):
        """
        @param source_dict  Dict containing State constructor parameters.
                            The keys "objs" and "grippers" are mandatory.
                            Refer to the documentation for additional format
                            instructions.
        @param type_config  dict mapping type names to block matrices
        @param grid_config dict to initialize the grids
        @return new State instance with the given attributes
        """
        if not isinstance(source_dict.get("objs"), dict) or \
                not isinstance(source_dict.get("grippers"), dict):
            raise ValueError(
                "source_dict must contain the keys 'objs' and 'grippers' "
                "mapping to dictionaries."
            )

        gc = None
        if grid_config:
            gc = grid_config
        if "grid_config" in source_dict:
            gc = GridConfig.from_dict(source_dict["grid_config"])
        if not gc:
            raise Exception("Either provide grid_config as a value in 'source_dict' or to 'from_dict' directly")

        try:
            # construct objects
            objs: Dict[int, T] = dict()
            for obj_id, obj_dict in source_dict["objs"].items():
                new_object = T.create_object(obj_dict, type_config)
                objs[new_object.id_n] = new_object

            # construct grippers
            grippers = dict()
            for gr_name, gr_dict in source_dict["grippers"].items():
                # get identifier or use gripper key (use str for consistency)
                id_n = gr_dict.get("id_n") or str(gr_name)
                new_gr = Gripper.from_dict(id_n, gr_dict)
                grippers[id_n] = new_gr

                # Not the nicest solution: Make sure any gripped object has
                # its 'gripped' attribute set to True
                if new_gr.gripped is not None:
                    objs[new_gr.gripped].gripped = True

            # construct targets
            targets = dict()
            if "targets" in source_dict:
                for obj_id, obj_dict in source_dict["targets"].items():
                    new_object = T.from_dict(obj_dict, type_config)
                    targets[new_object.id_n] = new_object

        except KeyError:
            raise KeyError(
                "Error during state initialization: JSON data "
                "does not have the right format.\n"
                "Please refer to the documentation."
            )
        state_id = None
        if "state_id" in source_dict:
            state_id = source_dict["state_id"]
        global_id = None
        if "global_id" in source_dict:
            global_id = source_dict["global_id"]
        return cls(gc, objs, grippers, targets, state_id, global_id)

    def remove_object(self, obj, object_is_target=False):
        """
        removes an object from the state dictionary and the grid
        @param obj  the object to remove
        @param object_is_target if True the object is a target
        """
        if object_is_target:
            self.targets.remove(obj)
            self.target_grid.remove_obj(obj)
        else:
            self.objects.remove(obj)
            self.object_grid.remove_obj(obj)

    def add_object(self, obj: T, object_is_target=False, check_position=False):
        """
        adds an object to the state dictionary and the grid
        @param obj  the object to add
        @param object_is_target if True the object is a target
        @param check_position: if True raises Exception when already occupied coords
        """
        if check_position and not self.is_legal_position(obj, object_is_target):
            raise Exception("Warning: Piece position is already occupied")
        # todo actually we also need to check if id already exists
        if object_is_target:
            self.target_grid.add_obj(obj)
            self.targets.add(obj)
        else:
            self.object_grid.add_obj(obj)
            self.objects.add(obj)

    def is_legal_position(self, obj: T, object_is_target=False):
        if object_is_target:
            return self.target_grid.is_legal_position(obj.occupied(), obj.id_n)
        return self.object_grid.is_legal_position(obj.occupied(), obj.id_n)

    def get_tile(self, x, y, obj_is_target=False):
        """
        returns the tile from a grid (object or target) at
        coordinates x, y
        """
        if obj_is_target is True:
            grid = self.target_grid
        else:
            grid = self.object_grid

        return grid.get_single_tile({"x": x, "y": y})

    def to_state_dict(self, include_grid_config=False):
        """
        Create a JSON-friendly representation of the current state
        @return dict containing current grippers and objects
        """
        state_dict = dict()
        state_dict["state_id"] = self.state_id
        if self.global_id:
            state_dict["global_id"] = self.global_id
        state_dict["grippers"] = self.grippers.to_json()
        state_dict["objs"] = self.objects.to_json()
        state_dict["targets"] = self.targets.to_json()
        if include_grid_config:
            state_dict["grid_config"] = self.object_grid.get_grid_config().to_dict()
        return state_dict
