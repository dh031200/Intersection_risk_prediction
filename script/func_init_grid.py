# %%
import numpy as np
import cv2
# %%
# background : (0,0,0)
# pedestrian : (78,154,6)
# car road: (128,128,128)
# crosswalk : (194,225,108)

seg_color = [[0, 0, 0], [6, 154, 78], [128, 128, 128], [108, 225, 194]]
seg_img = cv2.imread('../src/seg_map_v2_short.png')
# test_img = cv2.imread('../src/testimg.png')


class GridArray:
    def __init__(self, k, shape, seg_img):
        self.origin_shape = shape
        self.x, self.y = int(shape[1] / (k + 1)), int(shape[0] / (k + 1))
        self.grid_array = np.full((self.y, self.x, 1), 0, np.uint8)
        self.grid_seg_env = np.full((self.y, self.x, 1), 0, np.uint8)

        for y in range(self.y):
            for x in range(self.x):
                #print('y:', y)
                #print('x:', x)
                Y = y * (k + 1)
                X = x * (k + 1)
                unique, counts = np.unique(
                    seg_img[Y:Y + k, X:X + k], return_counts=True, axis=1)
                seg_num = list(unique[np.argmax(counts)][0])
                # cv2.imshow('point', seg_img[Y:Y + k, X:X + k])
                # test = cv2.rectangle(seg_img, (X, Y), (X + k, Y + k), (0, 0, 255), -1)
                # cv2.imshow('test', test)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     cv2.destroyAllWindows()
                self.grid_seg_env[y, x] = seg_color.index(seg_num)

    def update(self, coords, clss):
        grid_list = []

        for coord, cls in zip(coords, clss):
            if coord[1] < 0 or coord[1] >= self.origin_shape[1] or coord[0] < 0 or coord[0] >= self.origin_shape[0]:
                continue
            if cls == 'person':
                i_cls = 1
            elif cls == 'car':
                i_cls = 2
            else:
                continue

            y = int(coord[0] / (k + 1))
            x = int(coord[1] / (k + 1))
            #print('y: ', y)
            #print('x: ', x)

            self.grid_array[y, x] = i_cls

            grid_list.append((x, y))

        return grid_list
# %%


k = 15
shape = (745, 800, 3)

ga = GridArray(k, shape, seg_img)

# cv2.destroyAllWindows()

for i in ga.grid_seg_env:
    print(*i, sep=' ')

# %%
