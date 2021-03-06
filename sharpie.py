#!/usr/bin/env python
import cv2
import numpy as np
import skimage.segmentation as seg
import matplotlib.pyplot as plt
from skimage.util import img_as_float

#linear regression over super pixel groupings
def lin_reg(vals):
    n, s_x, s_y, s_xx, s_xy = vals[0],vals[1],vals[2],vals[3],vals[4]
    a = ((s_y * s_xx) - (s_x * s_xy)) / ((n * s_xx) - (s_x * s_x))
    b = ((n * s_xy) - (s_x * s_y)) / ((n * s_xx) - (s_x * s_x))
    return a,b


image_names = ['castle.jpeg']
for z in range(0,len(image_names)):
    #import image
    sharpie_test_o = img_as_float(cv2.imread(image_names[z]))
    #output image instantiation
    zero = np.empty_like(sharpie_test_o)
    #generate super pixels
    sharpie_test = seg.slic(sharpie_test_o,n_segments=1000,compactness = 0.00001,max_iter=30)
    h,w = sharpie_test.shape
    #x_min,x_max,y_min,y_max
    #defines super pixel boundaries
    x_y_min_max_values = [[-1,-1,-1,-1] for x in range(0,np.amax(sharpie_test)+1)]
    #linear regression sums
    n_sum_x_y_xx_xy = [[0.0,0.0,0.0,0.0,0.0] for x in range(0,np.amax(sharpie_test)+1)]
    for y in range(0,w):
        for x in range(0,h):
            #populate the values for linear regressions and super pixel bounds
            t = [1,x,y,x*x,x*y]
            for n in range(0,5):
                n_sum_x_y_xx_xy[sharpie_test[x][y]][n] += t[n]
            if x_y_min_max_values[sharpie_test[x][y]][0] == -1 or x_y_min_max_values[sharpie_test[x][y]][0] > x:
                x_y_min_max_values[sharpie_test[x][y]][0] = x
            if x_y_min_max_values[sharpie_test[x][y]][1] == -1 or x_y_min_max_values[sharpie_test[x][y]][1] < x:
                x_y_min_max_values[sharpie_test[x][y]][1] = x
            if x_y_min_max_values[sharpie_test[x][y]][2] == -1 or x_y_min_max_values[sharpie_test[x][y]][2] > y:
                x_y_min_max_values[sharpie_test[x][y]][2] = y
            if x_y_min_max_values[sharpie_test[x][y]][3] == -1 or x_y_min_max_values[sharpie_test[x][y]][3] < y:
                x_y_min_max_values[sharpie_test[x][y]][3] = y
    #plot the linear regressions for each super pixel
    for x in x_y_min_max_values:
        a,b = lin_reg(n_sum_x_y_xx_xy[x_y_min_max_values.index(x)])
        for xx in range(x[0],x[1]):
            if (a + xx*b) < 2447 and (a + xx*b) > 0:
                zero[xx][a + xx*b] = 255
    sharpie_test_i = zero + sharpie_test_o*255
    cv2.imwrite('res'+str(z)+'.png',np.rot90(zero,3))
    cv2.imwrite('img_and_res_'+str(z)+'.png',np.rot90(sharpie_test_i,3))
    print str(z)+" done"
