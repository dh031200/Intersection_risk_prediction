# %%
import numpy as np
import cv2
# %%
# background : (0,0,0)
# pedestrian : (78,154,6)
# car road: (128,128,128)
# crosswalk : (194,225,108)

class setGrid:

    def __init__(self, num_cells, virtual_img_size):
        self.num_cells = num_cells

        # width == height in all cases
        self.virtual_img_width = virtual_img_size[0] = 1000
        self.virtual_img_height = virtual_img_size[1] = 1000

        self.grid_array = np.full(
            (self.num_cells * self.num_cells), 0, np.uint8)

        self.grid_pixel_ratio = float(self.virtual_img_width / self.num_cells)

        # self.grid_index = 0

    def get_grid_from_coord(self, coords, clss):

        base_boundary = [0, 0, self.grid_pixel_ratio, self.grid_pixel_ratio]
        # print(base_boundary)
        # print(self.grid_pixel_ratio)

        # break_flag = False
        grid_index = 0

        rs_grid_list = []

        for coord, cls in zip(coords, clss):
            # print(coord)
            break_flag = False
            for horizon_shift_index in range(0, self.num_cells):
                target_X = coord[0] - \
                    self.grid_pixel_ratio * horizon_shift_index

                # if target_X < base_boundary[0]:
                #    grid_index = 0
                #    break

                if (base_boundary[0] <= target_X) and (target_X <= base_boundary[2]):

                    for vertical_shift_index in range(0, self.num_cells):
                        target_Y = coord[1] - \
                            self.grid_pixel_ratio * vertical_shift_index

                        if target_Y < base_boundary[1]:
                            grid_index = 0
                            break_flag = True

                        if (base_boundary[1] <= target_Y) and (target_Y <= base_boundary[3]):
                            grid_index = horizon_shift_index + vertical_shift_index * self.num_cells
                            # rs_grid_list.append(grid_index)
                            break_flag = True
                            break

                if break_flag:
                    break

            rs_grid_list.append(grid_index)

        return rs_grid_list
