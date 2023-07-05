from typing import Tuple

import numpy as np

from cogrip.constants import COLOR_NAME_TO_IDX, SHAPE_NAME_TO_IDX
from cogrip.pentomino.objects import Piece
from cogrip.pentomino.state import Board

COLORS_TO_NUMPY = {
    "gripper": np.array([100, 100, 100]),
    "white": np.array([255, 255, 255]),
    "red": np.array([255, 0, 0]),
    "orange": np.array([255, 165, 0]),
    "yellow": np.array([255, 255, 0]),
    "green": np.array([0, 128, 0]),
    "blue": np.array([0, 0, 255]),
    "cyan": np.array([0, 255, 255]),
    "purple": np.array([128, 0, 128]),
    "brown": np.array([139, 69, 19]),
    "grey": np.array([128, 128, 128]),
    "pink": np.array([255, 192, 203]),
    "olive green": np.array([128, 128, 0]),
    "navy blue": np.array([0, 0, 128]),
}


def to_rgb_array(board: Board, target_piece: Piece,
                 gripper_coords: Tuple[int, int], num_channels: int,
                 channel_first: bool, return_pixels: bool = True, padding: int = 0):
    assert num_channels in [3, 4, 5]
    max_size = board.grid_config.width

    # use pytorch as default (channels first)
    if return_pixels:
        fill_value = 255
    else:
        fill_value = 1
    rgb_array = np.full((3, max_size, max_size), fill_value=fill_value, dtype=np.uint8)  # fill with white

    if num_channels == 5:  # paint target on separate channel # fill with "transparent"
        target_channel = np.full((1, max_size, max_size), fill_value=fill_value, dtype=np.uint8)

    for y in range(max_size):
        for x in range(max_size):
            tile = board.object_grid.grid[y][x]
            if tile.objects:
                obj_id = tile.objects[0].id_n
                piece = board.get_piece(obj_id)
                symbol = piece.piece_config
                if return_pixels:
                    color_name = symbol.color.value_name
                    tile_color = COLORS_TO_NUMPY[color_name]
                    rgb_array[:, y, x] = tile_color
                else:
                    color_name = symbol.color.value_name
                    rgb_array[0, y, x] = COLOR_NAME_TO_IDX[color_name]
                    shape_name = symbol.shape.value
                    rgb_array[1, y, x] = SHAPE_NAME_TO_IDX[shape_name]
                if num_channels == 5 and target_piece.id_n == obj_id:
                    # reduce "transparency" on target pos; not ones b.c. out-of-world is zero
                    # thus the board would not be distinguishable form oow (but is that a problem?)
                    target_channel[:, y, x] = 0

    x, y = gripper_coords

    if num_channels == 3 and return_pixels:  # paint gripper directly on the canvas
        rgb_array[:, y, x] = COLORS_TO_NUMPY["gripper"]

    if num_channels >= 4:  # paint gripper on separate channel (also when 5 channels)
        alpha_channel = np.full((1, max_size, max_size), fill_value=255, dtype=np.uint8)  # fill with "transparent"
        alpha_channel[:, y, x] = 0  # reduce "transparency" on gripper pos; not ones b.c. out-of-world is zero
        rgb_array = np.concatenate([rgb_array, alpha_channel], axis=0)

    if num_channels >= 5:
        rgb_array = np.concatenate([rgb_array, target_channel], axis=0)

    if not channel_first:
        rgb_array = np.moveaxis(rgb_array, 0, -1)

    if padding > 0:
        padding = int(padding / 2)
        paddings = [(padding, padding)] * (len(rgb_array.shape) - 1)
        if paddings:
            if channel_first:
                paddings.insert(0, (0, 0))  # pre-pend no-padding
            else:
                paddings.append((0, 0))
            rgb_array = np.pad(rgb_array, mode="constant", pad_width=paddings)

    return rgb_array


def compute_fov(vision_obs, board: Board, gripper_coords: Tuple[int, int], fov_size: int, channel_first: bool):
    map_size = board.grid_config.width
    x, y = gripper_coords
    context_size = int((fov_size - 1) / 2)
    topx = x - context_size
    topy = y - context_size
    if channel_first:
        # fill with black or "out-of-world"
        num_channels = vision_obs.shape[0]
        view = np.full((num_channels, fov_size, fov_size), fill_value=0)
    else:
        # fill with black or "out-of-world"
        num_channels = vision_obs.shape[-1]
        view = np.full((fov_size, fov_size, num_channels), fill_value=0)
    for offy in range(fov_size):
        for offx in range(fov_size):
            vx = topx + offx
            vy = topy + offy
            if (vx >= 0 and vy >= 0) and (vx < map_size and vy < map_size):
                if channel_first:
                    arr = vision_obs[:, vy, vx]
                    view[:, offy, offx] = arr
                else:
                    arr = vision_obs[vy, vx]
                    view[offy, offx, :] = arr
    return view
