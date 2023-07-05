"""
The Mover module implements all movements
(flip, move, rotate) in a single function
All helpers function needed to make a movement
are also implemented here.
"""
from cogrip.core.obj import Obj


class Mover:
    def _gripper_can_move(self, gr_id, dx, dy, state):
        """
        check if a gripper can be moved on the grid
        if the movement type is not move this function
        will always return True as dx and dy will be zero
        """
        gripper_x, gripper_y = state.grippers.get_coords_for(gr_id)
        new_gr_pos = {
            "x": (gripper_x + dx),
            "y": (gripper_y + dy)
        }

        return state.object_grid.gripper_on_grid(new_gr_pos)

    def _get_new_coordinates(self, rotation_step, gr_obj, **kwargs):
        """
        based on the type of movement this function will
        return the new coordinates after the movement
        """
        d_angle = None

        if kwargs["type"] == "move":
            dx = kwargs["dx"]
            dy = kwargs["dy"]
            new_coordinates = gr_obj.occupied(
                gr_obj.x + dx, gr_obj.y + dy, gr_obj.block_matrix
            )
            new_matrix = gr_obj.block_matrix

        elif kwargs["type"] == "rotate":
            if kwargs.get("rotation_step") is None:
                step_size = rotation_step

            direction = kwargs["direction"]
            # determine the turning angle
            d_angle = direction * step_size

            # obtain rotated matrix
            new_matrix = Obj.rotate_block_matrix(
                gr_obj.block_matrix, d_angle
            )

            new_coordinates = gr_obj.occupied(
                gr_obj.x, gr_obj.y, new_matrix
            )

        elif kwargs["type"] == "flip":
            # obtain flipped matrix
            new_matrix = Obj.flip_block_matrix(
                gr_obj.block_matrix
            )

            new_coordinates = gr_obj.occupied(
                gr_obj.x, gr_obj.y, new_matrix
            )

        return new_coordinates, new_matrix, d_angle

    def _obj_on_target(self, obj, state):
        objs_on_target = list()
        for position in obj.occupied():
            # TODO: implement a function on grid side to get
            # converted coordinates from converter
            converted = state.object_grid.converter(position)
            for new_position in converted:
                tile = state.target_grid[new_position]
                if len(tile.objects) == 0:
                    # empty tile, return False
                    return False
                else:
                    objs_on_target += tile.objects

        # every position on target grid had only 1 element
        if len(set(objs_on_target)) == 1:
            target_id = objs_on_target[0]
            target_obj = state.targets[target_id]

            # object and target must have same form and color
            if target_obj.type == obj.type:
                if target_obj.color == obj.color:
                    return True

    def _is_legal_move(self, new_coordinates, gr_obj, state, lock_on_target):
        """
        check if the movement is allowed
        """
        # tiles are free and within limits
        obj_can_move = state.object_grid.is_legal_position(
            new_coordinates, gr_obj
        )

        # check if object is on a target
        if lock_on_target is True:
            on_target = self._obj_on_target(gr_obj)
        else:
            on_target = False

        return obj_can_move and not on_target

    def _move(self, gr_id, dx, dy, state):
        """
        move a gripper and the gripped object
        """
        state.move_gr(gr_id, dx, dy)
        state.move_obj(state.grippers.get_gripped_obj_for(gr_id), dx, dy)

    def _rotate(self, gr_obj_id, d_angle, state):
        """
        rotate an object
        """
        # update state
        state.rotate_obj(gr_obj_id, d_angle)

    def _flip(self, gr_obj_id, state):
        """
        flip an object
        """
        # update state
        state.flip_obj(gr_obj_id)

    def apply_movement(self, model, movement_type, gr_id, **kwargs):
        """
        this method applies a movement.
        Parameters:
            - Model instance
            - movement type {"move", "flip", "rotate"}
            - gr_id: id of the gripper

        based on the movement type the function expects
        keyword arguments:
            - move:     - x_steps
                        - y_steps

            - rotate:   - direction
                        - rotation_step (optional)

            - flip:     does not require extra arguments
        """
        # extract config and state from model
        config = model.config  # this should be like PhysicsConfig
        state = model.state

        # gripper only moves if we have a move type movement
        dx = 0
        dy = 0

        # calculate the distance if the movement is a move
        if movement_type == "move":
            step_size = state.grid_config.move_step
            # make dx and dy multiples of step_size
            dx = round(kwargs["x_steps"]) * step_size
            dy = round(kwargs["y_steps"]) * step_size

        # make sure gripper can move
        gripper_can_move = self._gripper_can_move(gr_id, dx, dy, state)

        if gripper_can_move:
            # check if gripper has an object
            gr_obj = state.grippers.get_gripped_obj_for(gr_id)
            if gr_obj:
                # obtain direction and rotation step
                # if nor present they will be initialized to None
                direction = kwargs.get("direction")
                rotation_step = kwargs.get("rotation_step")

                # obtain coordinates after movement
                movement_result = self._get_new_coordinates(
                    config.rotation_step,
                    gr_obj,
                    type=movement_type,
                    dx=dx,
                    dy=dy,
                    direction=direction,
                    rotation_step=rotation_step
                )
                new_coordinates, new_matrix, d_angle = movement_result

                # check if coordinates are legal
                good_move = self._is_legal_move(
                    new_coordinates, gr_obj, state, config.lock_on_target
                )

                # apply movement
                if good_move:
                    # remove object from grid
                    state.object_grid.remove_obj(gr_obj)

                    # apply movement according to type
                    if movement_type == "move":
                        self._move(gr_id, dx, dy, state)

                    elif movement_type == "flip":
                        self._flip(gr_obj.id_n, state)

                    elif movement_type == "rotate":
                        self._rotate(gr_obj.id_n, d_angle, state)

                    # add element to grid
                    state.object_grid.add_obj(gr_obj)

                    # print grid to terminal if verbose
                    if config.verbose is True:
                        print(state.object_grid)

            else:
                # only move the gripper
                state.move_gr(gr_id, dx, dy)
        return gripper_can_move
