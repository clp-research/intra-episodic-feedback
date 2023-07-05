import math

from cogrip.core.config import Config
from cogrip.core.gripper import Gripper
from cogrip.core.state import State
from cogrip.core.mover import Mover


class Engine:
    def __init__(self, config: Config):
        self.state = State(config)
        self.config = config
        self.mover = Mover()

    # --- getter --- #

    def get_obj_by_id(self, obj_id):
        return self.state.objects[obj_id]

    def get_gripper_by_id(self, gr_id):
        """
        @return Gripper instance or None if id is not registered
        """
        return self.state.grippers[gr_id]

    def get_gripped_obj(self, gr_id):
        return self.state.grippers.get_gripped_obj_for(gr_id)

    def get_gripper_coords(self, gr_id):
        """
        @return list: [x-coordinate, y-coordinate]
        """
        return self.state.grippers.get_coords_for(gr_id)

    def get_config(self):
        return self.config.to_dict()

    def get_width(self):
        return self.state.grid_config.width

    def get_height(self):
        return self.state.grid_config.height

    def get_type_config(self):
        return self.config.type_config

    # --- Set up and configuration --- #

    def set_state(self, state):
        """
        Initialize the model's (game) state.
        @param state	json file name or dict (e.g., parsed json)
                        or State instance
        """
        if isinstance(state, str):
            self.state = State.from_json(
                state, self.get_type_config(), self.config
            )
        elif isinstance(state, dict):
            self.state = State.from_state_dict(
                state, self.get_type_config(), self.config
            )
        elif isinstance(state, State):
            self.state = state
        else:
            raise TypeError("Parameter state must be a json file name, "
                            "dict, or State instance.")

    def set_config(self, config):
        """
        Change the model's configuration. Overwrites any attributes
        passed in config and leaves the rest as before. New keys simply added.
        @param config	json file name or dict (e.g., parsed json)
                        or Config instance
        """
        # config is a JSON string or parsed JSON dictionary
        if isinstance(config, str):
            self.config = Config.from_json(config)
        elif isinstance(config, dict):
            self.config = Config.from_dict(config)
        elif isinstance(config, Config):
            self.config = config
        else:
            raise TypeError("Parameter config must be a json file name, "
                            "dict, or Config instance")

    def reset(self):
        """
        Reset the current state.
        """
        self.state = State(self.config)

    # --- Gripper manipulation --- #
    def add_gr(self, gr_id, start_x: int = None, start_y: int = None):
        """
        Add a new gripper to the internal state.
        The start position is the center. Notifies listeners.
        @param gr_id 	identifier for the new gripper
        @param start_x  starting x coord
        @param start_y  starting y coord
        """
        if start_x is None:
            start_x = self.get_width() / 2
        if start_y is None:
            start_y = self.get_height() / 2
        # if a new gripper was created, notify listeners
        if gr_id not in self.state.grippers:
            self.state.grippers.add(Gripper(gr_id, start_x, start_y))

    def remove_gr(self, gr_id):
        """
        Delete a gripper from the internal state and notify listeners.
        @param gr_id 	identifier of the gripper to remove
        """
        if gr_id in self.state.grippers:
            self.state.grippers.remove(gr_id)

    def can_ungrip(self, obj):
        """
        This function decides wether a block can be ungripped
        if snap_to_grid is on and the object lies between
        lines (coordinates are float), it will automatically
        try to find a free spot and place the object there
        """
        # without snap to grid a gripper can always ungrip
        if self.config.snap_to_grid is False:
            return True

        # integer positions are always plotted on the grid
        if float(obj.x).is_integer() and float(obj.y).is_integer():
            return True
        else:
            # x or y not on grid
            possible_positions = [
                (math.ceil(obj.x), math.ceil(obj.y)),
                (math.ceil(obj.x), math.floor(obj.y)),
                (math.floor(obj.x), math.ceil(obj.y)),
                (math.floor(obj.x), math.floor(obj.y))
            ]

            for new_x, new_y in possible_positions:
                occupied = obj.occupied(new_x, new_y)
                if self.mover._is_legal_move(
                        occupied,
                        obj,
                        self.state,
                        self.config.lock_on_target):
                    # move object

                    # 1 - remove obj from state
                    self.state.remove_object(obj)

                    # 2 - change x and y in object
                    obj.x = new_x
                    obj.y = new_y

                    # 3 - add object to state
                    self.state.add_object(obj)
                    return True

            # if no nearby position if free, cannot place it
            return False

    def grip(self, gr_id):
        """
        Attempt a grip / ungrip.
        @param gr_id 	gripper id
        """
        # if some object is already gripped, ungrip it
        old_gripped = self.get_gripped_obj(gr_id)
        if old_gripped:
            allowed = self.can_ungrip(old_gripped)
            if allowed:
                # state takes care of detaching object and gripper
                self.state.ungrip(gr_id)
        else:
            # Check if gripper hovers over some object
            new_gripped = self._get_grippable(gr_id)
            # changes to object and gripper
            if new_gripped is not None:
                self.state.grip(gr_id, new_gripped)

    def _get_grippable(self, gr_id):
        """
        Find an object that is in the range of the gripper.
        @param id 	gripper id
        @return id of object to grip or None
        """
        # Gripper position. It is just a point.
        x, y = self.get_gripper_coords(gr_id)

        tile = self.state.get_tile(x, y)
        # if there is an object on tile, return last object
        if tile.objects:
            return tile.objects[-1].id_n
        return None
