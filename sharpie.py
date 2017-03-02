#!/usr/bin/env python
import cv2
import numpy as np
import skimage.segmentation as seg
import matplotlib.pyplot as plt
from skimage.util import img_as_float

def lin_reg(vals):
    n, s_x, s_y, s_xx, s_xy = vals[0],vals[1],vals[2],vals[3],vals[4]
    a = ((s_y * s_xx) - (s_x * s_xy)) / ((n * s_xx) - (s_x * s_x))
    b = ((n * s_xy) - (s_x * s_y)) / ((n * s_xx) - (s_x * s_x))
    return a,b


image_names = 'IMG_4025.JPG','IMG_4026.JPG','IMG_4027.JPG','IMG_4028.JPG'
for z in range(0,4):
    sharpie_test_o = img_as_float(cv2.imread(image_names[z]))
    zero = np.zeros_like(sharpie_test_o)
    sharpie_test = seg.slic(sharpie_test_o,n_segments=10000,compactness = 0.00001,max_iter=20)
    h,w = sharpie_test.shape
    #x_min,x_max,y_min,y_max
    x_y_min_max_values = [[-1,-1,-1,-1] for x in range(0,np.amax(sharpie_test)+1)]
    n_sum_x_y_xx_xy = [[0.0,0.0,0.0,0.0,0.0] for x in range(0,np.amax(sharpie_test)+1)]
    for y in range(0,h):
        for x in range(0,w):
            n_sum_x_y_xx_xy[sharpie_test[x][y]][0] += 1
            n_sum_x_y_xx_xy[sharpie_test[x][y]][1] += x
            n_sum_x_y_xx_xy[sharpie_test[x][y]][2] += y
            n_sum_x_y_xx_xy[sharpie_test[x][y]][3] += x*x
            n_sum_x_y_xx_xy[sharpie_test[x][y]][4] += x*y
            if x_y_min_max_values[sharpie_test[x][y]][0] == -1 or x_y_min_max_values[sharpie_test[x][y]][0] > x:
                x_y_min_max_values[sharpie_test[x][y]][0] = x
            if x_y_min_max_values[sharpie_test[x][y]][1] == -1 or x_y_min_max_values[sharpie_test[x][y]][1] < x:
                x_y_min_max_values[sharpie_test[x][y]][1] = x
            if x_y_min_max_values[sharpie_test[x][y]][2] == -1 or x_y_min_max_values[sharpie_test[x][y]][2] > y:
                x_y_min_max_values[sharpie_test[x][y]][2] = y
            if x_y_min_max_values[sharpie_test[x][y]][3] == -1 or x_y_min_max_values[sharpie_test[x][y]][3] < y:
                x_y_min_max_values[sharpie_test[x][y]][3] = y
    for x in x_y_min_max_values:
        a,b = lin_reg(n_sum_x_y_xx_xy[x_y_min_max_values.index(x)])
        for xx in range(x[0],x[1]):
            if (a + xx*b) < 2447 and (a + xx*b) > 0:
                zero[xx][a + xx*b] = 1
    #sharpie_test_i = seg.mark_boundaries(zero,sharpie_test,color = (0,1,0),mode='thick')*255
    cv2.imwrite('img'+str(z)+'.png',np.rot90(zero,3)*255)
