from typing import List, Dict, Tuple

import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import colors as plt_colors

from cogrip.pentomino.objects import Piece
from cogrip.pentomino.symbolic.types import SymbolicPiece
from cogrip.core.grid import GridConfig
from cogrip.core.state import State


class BoardPlotContext:

    def __init__(self, image_size: Tuple[int]):
        fig_size = (image_size[0] / 100, image_size[1] / 100)
        self.fig, self.ax = plt.subplots(figsize=fig_size, dpi=100)
        self.image_context = None
        self.ax.tick_params(axis="both", which="both", bottom=False, top=False, left=False, right=False,
                            labelbottom=False, labelleft=False)
        self.ax.axis('off')

    def draw_board(self, np_board, obj_colors, bounds, verbose=False):
        ax = self.ax
        if verbose:
            print("Get borders and eliminate seams within them")
        borders = self.__get_borders(np_board)
        l_borders = self.__find_long(borders)
        if verbose:
            print("Plot borders of all objects")
        for x, y in l_borders:
            ax.plot(x, y, scaley=False, linestyle="-", linewidth=.5, color="black")
        if verbose:
            print("Plot image")
        cmap = plt_colors.ListedColormap([o[1] for o in obj_colors])
        norm = plt_colors.BoundaryNorm(bounds, cmap.N)
        if self.image_context:
            self.image_context.set_data(np_board)
        else:
            # lazy init of the image_context for later re-use
            self.image_context = ax.imshow(np_board)
        self.image_context.set_cmap(cmap)
        self.image_context.set_norm(norm)
        if verbose:
            print("remove labels and ticks on edges")
        if verbose:
            print("Draw Canvas")
        # This must happen here otherwise we have color artifacts
        self.fig.tight_layout(pad=0)
        ax.figure.canvas.draw()
        if verbose:
            print("Get the RGBA buffer from the figure")
        w, h = self.fig.canvas.get_width_height()
        buf = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(w, h, 3)
        del cmap
        del norm
        ax.lines.clear()  # remove borders for next plot
        return buf

    def close(self):
        plt.cla()
        plt.clf()
        plt.close(self.fig)

    def __get_borders(self, np_board: np.array, to_add=0.5):
        """ iterate over the the 2D array (from get_matrix) and detect borders """
        lines = list()
        for y, row in enumerate(np_board):
            for x, tile in enumerate(row):
                if tile != 0:
                    output_list = lines
                    # upper bound
                    if y == 0 or np_board[y - 1][x] != tile:
                        this_line = (
                            (x - to_add, x + to_add),
                            (y - to_add, y - to_add),
                        )
                        output_list.append(this_line)
                    # lower bound
                    if y == len(np_board) - 1 or np_board[y + 1][x] != tile:
                        this_line = (
                            (x - to_add, x + to_add),
                            (y + to_add, y + to_add),
                        )
                        output_list.append(this_line)
                    # right bound
                    if x == len(np_board[0]) - 1 or np_board[y][x + 1] != tile:
                        this_line = (
                            (x + to_add, x + to_add),
                            (y - to_add, y + to_add),
                        )
                        output_list.append(this_line)
                    # left bound
                    if x == 0 or np_board[y][x - 1] != tile:
                        this_line = (
                            (x - to_add, x - to_add),
                            (y - to_add, y + to_add),
                        )
                        output_list.append(this_line)
        return lines

    def __find_long(self, coord):
        """
        reconstruct long borders to avoid seam lines
        in objects black borders
        """
        h_lines = dict()
        v_lines = dict()
        results = list()

        # save lines in dictionaries
        for x, y in coord:
            # vertical line
            if len(set(x)) == 1:
                if x[0] not in v_lines:
                    v_lines[x[0]] = set()
                v_lines[x[0]].add(y)

            # horizontal line
            elif len(set(y)) == 1:
                if y[0] not in h_lines:
                    h_lines[y[0]] = set()
                h_lines[y[0]].add(x)
        # reconstruct longest lines from each entry
        # in horizontal dictionary
        for y, complete_line in h_lines.items():
            complete_line = list(complete_line)
            complete_line.sort()
            begin, end = complete_line[0]
            i = 1
            if len(complete_line) == 1:
                results.append((complete_line[0], (y, y)))
            else:
                # iterate over the list and reconstruct the longest lines
                while i < len(complete_line):
                    this_b, this_e = complete_line[i]
                    if end == this_b:
                        # lines continues, update end
                        end = this_e
                        # last element, save
                        if i == len(complete_line) - 1:
                            results.append(((begin, end), (y, y)))
                    else:
                        # end of line
                        results.append(((begin, end), (y, y)))
                        begin, end = complete_line[i]
                    i += 1
        # reconstruct longest lines from each entry
        # in horizontal dictionary
        for x, complete_line in v_lines.items():
            complete_line = list(complete_line)
            complete_line.sort()
            begin, end = complete_line[0]
            i = 1
            if len(complete_line) == 1:
                results.append(((x, x), complete_line[0]))
            else:
                # iterate over the list and reconstruct the longest lines
                while i < len(complete_line):
                    this_b, this_e = complete_line[i]
                    if end == this_b:
                        # lines continues
                        end = this_e
                        # last element, save
                        if i == len(complete_line) - 1:
                            results.append(((x, x), (begin, end)))
                    else:
                        # end of line
                        results.append(((x, x), (begin, end)))
                        begin, end = complete_line[i]
                    i += 1
        return results


class Board(State[Piece]):

    def __init__(self, grid_config: GridConfig, board_id=None):
        super(Board, self).__init__(grid_config, state_id=board_id)

    #
    def to_rgb_array(self):
        color_grid = []
        for row in self.object_grid:
            color_row = []
            for tile in row:
                tile_color = [255, 255, 255]  # white
                if tile.objects:
                    obj_id = tile.objects[0].id_n
                    tile_color = self.get_piece(obj_id).piece_config.color.value_rgb
                color_row.append(tile_color)
            color_grid.append(color_row)
        return np.array(color_grid)

    def get_piece(self, piece_id: int) -> Piece:
        return self.objects[piece_id]

    def add_piece(self, piece: Piece, max_attempts=100, verbose=False):
        for attempt in range(max_attempts):  # re-create with potentially different coords
            if self.is_legal_position(piece):
                self.add_object(piece, check_position=False)
                return True
        if verbose:
            print(f"Max attempts reached, cannot add piece from {piece.piece_config}")
        return False

    def add_pieces_from_symbols(self, piece_configs: List[SymbolicPiece], max_attempts=100):
        all_success = True
        for piece_config in piece_configs:
            if not self.add_piece_from_symbol(piece_config, max_attempts):
                all_success = False
        return all_success

    def add_piece_from_symbol(self, piece_symbol: SymbolicPiece, max_attempts=100, verbose=False):
        for attempt in range(max_attempts):  # re-create with potentially different coords
            piece = Piece.from_symbol(len(self.objects), piece_symbol, self.object_grid.width, self.object_grid.height)
            if self.is_legal_position(piece):
                self.add_object(piece, check_position=False)
                return True
        if verbose:
            print(f"Max attempts reached, cannot add piece from {piece_symbol}")
        return False

    def to_image_array(self, image_size, ctx: BoardPlotContext = None, verbose=False, force_headless=False):
        """ convert a state to an RGB numpy array """
        if verbose:
            print("Get Matrix")
        np_board, obj_colors, color_dict, bounds = self.__get_matrix()
        if verbose:
            print("Create Figure")
        if ctx:
            return ctx.draw_board(np_board, obj_colors, bounds)
        if force_headless:
            backend = matplotlib.get_backend()
            matplotlib.use('Agg')  # headless
        ctx = BoardPlotContext(image_size)
        image = ctx.draw_board(np_board, obj_colors, bounds)
        ctx.close()
        if force_headless:
            matplotlib.use(backend, force=True)  # switch back
        return image

    def __get_matrix(self) -> (np.array, List, Dict, List):
        """
        given a state, a 2D numpy array is created where each object and target
        is identified by a unique number which is then used to assign
        it its colour. this is then used to generate a heatmap which represents
        the plotted state.
        """
        x_dim = self.object_grid.width * self.object_grid.converter.multiplier
        y_dim = self.object_grid.height * self.object_grid.converter.multiplier
        # initialize variables to plot the grid
        data = np.zeros((x_dim, y_dim))
        obj_colors = [("white", "#FFFFFF", [255, 255, 255])]
        bounds = [-10, 0.5]
        color_dict = {0: ("white", "#FFFFFF", [255, 255, 255])}
        val = 1
        # plot objects
        val += 1
        for i, obj in enumerate([piece for piece in self.objects]):
            val += i
            bounds.append(val + 0.5)
            obj_colors.append(obj.color)
            color_dict[val] = obj.color
            self.__draw_obj(obj, data, val)
        return data, obj_colors, color_dict, bounds

    def __draw_obj(self, obj: Piece, data: np.array, fill_value: int):
        """
        draw a single object on a 2D grid with a unique fillvalue
        """
        for y, row in enumerate(obj.block_matrix):
            for x, tile in enumerate(row):
                if tile == 1:
                    # convert coordinates with converter!
                    for new_c in self.object_grid.converter({"x": x + obj.x, "y": y + obj.y}):
                        this_x = new_c["x"]
                        this_y = new_c["y"]

                        grid_x = int(this_x)
                        grid_y = int(this_y)
                        data[grid_y][grid_x] = fill_value

    def get_bbox(self, image_width: int, image_height: int, piece: Piece):
        """
        given the RGB np.array of the state and the object dictionary,
        slice the state to create a small image of size len(block_matrix)
        """
        x_factor = image_width / self.object_grid.width
        y_factor = image_height / self.object_grid.height
        obj = piece
        min_y = len(obj.block_matrix)
        min_x = len(obj.block_matrix[0])
        max_y = 0
        max_x = 0
        for y, row in enumerate(obj.block_matrix):
            for x, tile in enumerate(row):
                if tile == 1:
                    # update upper-left corner
                    if y < min_y:
                        min_y = y
                    if x < min_x:
                        min_x = x
                    # update upper-right corner
                    if y > max_y:
                        max_y = y
                    if x > max_x:
                        max_x = x
        this_x = (obj.x + min_x) * x_factor
        this_y = (obj.y + min_y) * y_factor
        x_limit = (obj.x + max_x + 1) * x_factor
        y_limit = (obj.y + max_y + 1) * y_factor
        return int(this_x), int(x_limit), int(this_y), int(y_limit)

    @classmethod
    def from_state_dict(cls, state_dict, grid_config: GridConfig = None, check_piece_position: bool = False):
        board_id = None
        if "state_id" in state_dict:
            board_id = state_dict["state_id"]
        if grid_config is None and "grid_config" in state_dict:
            grid_config = GridConfig.from_dict(state_dict["grid_config"])
        if grid_config is None:
            raise Exception("Provide grid_config either as argument or as state_dict entry")
        board = Board(grid_config, board_id)
        for obj in state_dict["objs"].values():  # this is weirdly enough a dict
            piece = Piece.from_dict(obj, grid_config)
            board.add_piece(piece, check_piece_position)
        return board
