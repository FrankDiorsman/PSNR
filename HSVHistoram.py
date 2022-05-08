# -*- coding:utf-8 -*-
# 本程序用于将一张彩色图片分解HSV分量显示，并显示直方图
import cv2  # 导入opencv模块
import numpy as np
import matplotlib.pyplot as plt


# 绘制直方图函数
def grayHist(img, name):
    h, w = img.shape[:2]
    pixelSequence = img.reshape([h * w, ])
    numberBins = 256
    histogram, bins, patch = plt.hist(pixelSequence, numberBins,
                                      facecolor='black', histtype='bar')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel("灰度值")
    plt.ylabel("像素数量")
    plt.axis([0, 255, 0, np.max(histogram)])
    #plt.savefig("D:/HSV-deal-images/" + name + ".png")
    plt.show()

# 绘制直方图欧氏距离
def hsv_ou_distance(realhsvChannels, shotcuthsvChannels):
    m = 0
    n = 0
    m = len(realhsvChannels[0])
    # print(m)
    n = len(shotcuthsvChannels[0])
    # print(n)

    hist1_B = cv2.calcHist([realhsvChannels[0]], [0], None, [256], [0, 255])
    hist2_B = cv2.calcHist([shotcuthsvChannels[0]], [0], None, [256], [0, 255])
    hist1_G = cv2.calcHist([realhsvChannels[1]], [0], None, [256], [0, 255])
    hist2_G = cv2.calcHist([shotcuthsvChannels[1]], [0], None, [256], [0, 255])
    hist1_R = cv2.calcHist([realhsvChannels[2]], [0], None, [256], [0, 255])
    hist2_R = cv2.calcHist([shotcuthsvChannels[2]], [0], None, [256], [0, 255])

    for i in range(1, 255):
        hist1_B[i] = hist1_B[i] / m
        hist2_B[i] = hist2_B[i] / n
        hist1_G[i] = hist1_G[i] / m
        hist2_G[i] = hist2_G[i] / n
        hist1_R[i] = hist1_R[i] / m
        hist2_R[i] = hist2_R[i] / n
    oud_B = 0
    oud_G = 0
    oud_R = 0
    for i in range(1, 255):
        oud_B = oud_B + pow((hist1_B[i] - hist2_B[i]), 2)
        oud_G = oud_G + pow((hist1_G[i] - hist2_G[i]), 2)
        oud_R = oud_R + pow((hist1_R[i] - hist2_R[i]), 2)
    oud_B = np.sqrt(oud_B)
    oud_G = np.sqrt(oud_G)
    oud_R = np.sqrt(oud_R)
    return oud_B, oud_G, oud_R


img = cv2.imread("I:/DL/HOG/real.jpg")  # 导入图片，图片放在程序所在目录
 cv2.namedWindow("imagshow", 2)  # 创建一个窗口
cv2.imshow('imagshow', img)  # 显示原始图片

# 使用cvtColor转换为HSV图
realout_img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转换为灰度图
realhsvChannels = cv2.split(realout_img_HSV)  # 将HSV格式的图片分解为3个通道

cv2.namedWindow("Hue", 2)  # 创建一个窗口
cv2.imshow('Hue', realhsvChannels[0])  # 显示Hue分量
grayHist(realhsvChannels[0], "H-Histogram")
cv2.namedWindow("Saturation", 2)  # 创建一个窗口
cv2.imshow('Saturation', realhsvChannels[1])  # 显示Saturation分量
grayHist(realhsvChannels[1], "S-Histogram")
cv2.namedWindow("Value", 2)  # 创建一个窗口
cv2.imshow('Value', realhsvChannels[2])  # 显示Value分量
grayHist(realhsvChannels[2], "V-Histogarm")

img = cv2.imread("I:/DL/HOG/shotcut.jpg")  # 导入图片，图片放在程序所在目录
cv2.namedWindow("imagshow", 2)  # 创建一个窗口
cv2.imshow('imagshow', img)  # 显示原始图片

# 使用cvtColor转换为HSV图
shotcut_out_img_HSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # 将图片转换为灰度图
shotcuthsvChannels = cv2.split(shotcut_out_img_HSV)  # 将HSV格式的图片分解为3个通道

cv2.namedWindow("Hue", 2)  # 创建一个窗口
cv2.imshow('Hue', shotcuthsvChannels[0])  # 显示Hue分量
grayHist(shotcuthsvChannels[0], "H-Histogram")
cv2.namedWindow("Saturation", 2)  # 创建一个窗口
cv2.imshow('Saturation', shotcuthsvChannels[1])  # 显示Saturation分量
grayHist(shotcuthsvChannels[1], "S-Histogram")
cv2.namedWindow("Value", 2)  # 创建一个窗口
cv2.imshow('Value', shotcuthsvChannels[2])  # 显示Value分量
grayHist(shotcuthsvChannels[2], "V-Histogarm")

# 1.巴氏距离
match1 = cv2.compareHist(realout_img_HSV, shotcut_out_img_HSV, cv2.HISTCMP_BHATTACHARYYA)
# 2.相关性
match2 = cv2.compareHist(realout_img_HSV, shotcut_out_img_HSV, cv2.HISTCMP_CORREL)
print("巴氏距离：%s, 相关性: %s" % (match1, match2))

resultH, resultS, resultV = hsv_ou_distance(realhsvChannels,shotcuthsvChannels)
print('H+S+V Color-ou-distance:',resultH/255, resultS/255, resultV/255)

cv2.waitKey(0)  # 等待用户操作
