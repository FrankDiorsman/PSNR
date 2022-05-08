# -*- coding:utf-8 -*-
# Linda Li 2019/8/25 15:44 cv_28_图像直方图 PyCharm

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt


def plot_demo(image):
    """
    image.ravel()统计频次的
    bins 256,256条直方
    range[0,256]
    """
    plt.hist(image.ravel(), 256, [0, 256])
    plt.show()


print("-------hello python--------")
src = cv.imread("I:/DL/HOG/shotcut.jpg")
cv.namedWindow("input image", cv.WINDOW_AUTOSIZE)
cv.imshow("input image", src)

plot_demo(src)

cv.waitKey(0)
cv.destroyAllWindows()



