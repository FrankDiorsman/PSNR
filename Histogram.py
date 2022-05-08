# -*- coding: utf-8 -*-
# !/usr/bin/python

import cv2
import numpy as np
from matplotlib import pyplot as plt

from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

gray_level = 256  # 灰度级

# 最简单的以灰度直方图作为相似比较的实现
def classify_gray_hist(image1, image2, size=(1024, 1024)):
    # 先计算直方图
    # 几个参数必须用方括号括起来
    # 这里直接用灰度图计算直方图，所以是使用第一个通道，
    # 也可以进行通道分离后，得到多个通道的直方图
    # bins 取为16
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 可以比较下直方图
    plt.plot(range(256), hist1, 'r')
    plt.plot(range(256), hist2, 'b')
    # plt.show()
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

# 计算单通道的直方图的相似值
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # 计算直方图的重合度
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree

# 通过得到每个通道的直方图来计算相似度
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # 将图像resize后，分离为三个通道，再计算每个通道的相似值
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data

# 平均哈希算法计算
def classify_aHash(image1, image2):
    image1 = cv2.resize(image1, (8, 8))
    image2 = cv2.resize(image2, (8, 8))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hash1 = getHash(gray1)
    hash2 = getHash(gray2)
    return Hamming_distance(hash1, hash2)

def classify_pHash(image1, image2):
    image1 = cv2.resize(image1, (32, 32))
    image2 = cv2.resize(image2, (32, 32))
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    # 将灰度图转为浮点型，再进行dct变换
    dct1 = cv2.dct(np.float32(gray1))
    dct2 = cv2.dct(np.float32(gray2))
    # 取左上角的8*8，这些代表图片的最低频率
    # 这个操作等价于c++中利用opencv实现的掩码操作
    # 在python中进行掩码操作，可以直接这样取出图像矩阵的某一部分
    dct1_roi = dct1[0:8, 0:8]
    dct2_roi = dct2[0:8, 0:8]
    hash1 = getHash(dct1_roi)
    hash2 = getHash(dct2_roi)
    return Hamming_distance(hash1, hash2)

# 输入灰度图，返回hash
def getHash(image):
    avreage = np.mean(image)
    hash = []
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i, j] > avreage:
                hash.append(1)
            else:
                hash.append(0)
    return hash

# 计算汉明距离
def Hamming_distance(hash1, hash2):
    num = 0
    for index in range(len(hash1)):
        if hash1[index] != hash2[index]:
            num += 1
    return num

def probability_to_histogram(img, prob):
    prob = np.cumsum(prob)  # 累计概率
    img_map = [int(i * prob[i]) for i in range(256)]  # 像素值映射
   # 像素值替换
    assert isinstance(img, np.ndarray)
    r, c = img.shape
    for ri in range(r):
        for ci in range(c):
            img[ri, ci] = img_map[img[ri, ci]]

    return img

def pixel_probability(img):
    assert isinstance(img, np.ndarray)
    prob = np.zeros(shape=(256))
    for rv in img:
        for cv in rv:
            prob[cv] += 1
    r, c = img.shape
    prob = prob / (r * c)
    return prob

def plot(y, name):
    plt.figure(num=name)
    plt.bar([i for i in range(gray_level)], y, width=1)

def gray_ou_distance(img1, img2):
    hist1 = cv2.calcHist([img1], [0], None, [256], [0, 255])
    hist2 = cv2.calcHist([img2], [0], None, [256], [0, 255])
    m = 0
    n = 0
    m = len(img1.nonzero()[0])
    #print(m)
    n = len(img2.nonzero()[0])
    #print(n)

    for i in range(1, 255):
        hist1[i] = hist1[i] / m
        hist2[i] = hist2[i] / n
    oud = 0
    for i in range(1, 255):
        oud = oud + pow((hist1[i] - hist2[i]), 2)
    oud = np.sqrt(oud)
    return oud

def Color_ou_distance(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    m = 0
    n = 0
    m = len(img1_gray.nonzero()[0])
    #print(m)
    n = len(img2_gray.nonzero()[0])
    #print(n)

    img1_B = img1[:, :, 0]
    img1_G = img1[:, :, 1]
    img1_R = img1[:, :, 2]

    img2_B = img2[:, :, 0]
    img2_G = img2[:, :, 1]
    img2_R = img2[:, :, 2]

    hist1_B = cv2.calcHist([img1_B], [0], None, [256], [0, 255])
    hist2_B = cv2.calcHist([img2_B], [0], None, [256], [0, 255])
    hist1_G = cv2.calcHist([img1_G], [0], None, [256], [0, 255])
    hist2_G = cv2.calcHist([img2_G], [0], None, [256], [0, 255])
    hist1_R = cv2.calcHist([img1_R], [0], None, [256], [0, 255])
    hist2_R = cv2.calcHist([img2_R], [0], None, [256], [0, 255])

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

def create_rgb_hist(image):
    h, w, c = image.shape
    rgbHist = np.zeros([16*16*16, 1], np.float32)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            b = image[row, col, 0]
            g = image[row, col, 1]
            r = image[row, col, 2]
            index = int(b/bsize)*16*16 + int(g/bsize)*16 + int(r/bsize)
            rgbHist[int(index), 0] = rgbHist[int(index), 0] + 1
    return rgbHist

def create_HSV_hist(image):
    h, w, c = image.shape
    hsvHist = np.zeros([16*16*16, 1], np.float32)
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bsize = 256 / 16
    for row in range(h):
        for col in range(w):
            h = img_hsv[row, col, 0]
            s = img_hsv[row, col, 1]
            v = img_hsv[row, col, 2]
            index = int(v/bsize)*16*16 + int(s/bsize)*16 + int(h/bsize)
            hsvHist[int(index), 0] = hsvHist[int(index), 0] + 1
    return hsvHist

def hist_compare(image1, image2):
    hist1 = create_rgb_hist(image1)
    hist2 = create_rgb_hist(image2)
    match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    match3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    print("巴氏距离: %s, 相关性: %s, 卡方: %s" % (match1, match2, match3))

if __name__ == '__main__':
    imgreal = cv2.imread('I:/DL/HOG/real.jpg')
    imgshotcut = cv2.imread('I:/DL/HOG/shotcut.jpg')
    imgs = np.hstack([imgreal, imgshotcut])
    cv2.imshow('imgreal', imgs)
    cv2.waitKey()

    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv2.calcHist([imgreal], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlabel('亮度值')
        plt.ylabel('像素数')
        plt.xlim([0, 256])
    plt.legend(('蓝色分量', '绿色分量', '红色分量'), loc='best')

    color = ("blue", "green", "red")
    for i, color in enumerate(color):
        hist = cv2.calcHist([imgshotcut], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
        plt.xlabel('亮度值')
        plt.ylabel('像素数')
        plt.xlim([0, 256])
    plt.legend(('蓝色分量', '绿色分量', '红色分量'), loc='best')

    hist_compare(imgreal, imgshotcut)

    resultB, resultG, resultR = Color_ou_distance(imgreal, imgshotcut)
    print('B+G+R Color-ou-distance:',resultB, resultG, resultR)

    imgreal = cv2.imread('I:/DL/HOG/real.jpg', 0)
    imgshotcut = cv2.imread('I:/DL/HOG/shotcut.jpg', 0)
    imgs = np.hstack([imgreal, imgshotcut])
    cv2.imshow('imgreal', imgs)
    cv2.waitKey()

    grayresult = gray_ou_distance(imgreal,imgshotcut)
    print('gray_ou_distance:',grayresult)

    probreal = pixel_probability(imgreal)
    probshotcut = pixel_probability(imgshotcut)
    plot(probreal, "原图直方图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("原始拍摄图像灰度直方图")
    plt.xlabel("灰度级")
    plt.ylabel("灰度值")
    plot(probshotcut, "截图直方图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("渲染成像图像灰度直方图")
    plt.xlabel("灰度级")
    plt.ylabel("灰度值")

    # 直方图均衡化
    imgreal = probability_to_histogram(imgreal, probreal)
    imgshotcut = probability_to_histogram(imgshotcut, probshotcut)
    probreal = pixel_probability(imgreal)
    probshotcut = pixel_probability(imgshotcut)

    cv2.imwrite("source_hist_real.jpg", imgreal)  # 保存图像
    cv2.imwrite("source_hist_shotcut.jpg", imgshotcut)  # 保存图像
    plot(probreal, "原图均衡化直方图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("原始拍摄图像灰度直方图")
    plt.xlabel("灰度级")
    plt.ylabel("灰度值")
    plot(probshotcut, "截图均衡化直方图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("渲染成像图像灰度直方图")
    plt.xlabel("灰度级")
    plt.ylabel("灰度值")
    plt.show()

    print('互相关直方图')
    hisarray = np.correlate(probreal, probshotcut, 'full')
    n, bins,patches = plt.hist(hisarray,range = [0,0.009])
    #plot(imgshotcut, "互相关直方图")
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.title("互相关灰度直方图")
    plt.xlabel("灰度级")
    plt.ylabel("灰度值")
    plt.show()
    max_index, max_number = max(enumerate(hisarray))
    print(max_number,max_index)

    imgss = np.hstack([imgreal,imgshotcut])
    cv2.imshow('imgreal', imgss)
    cv2.waitKey()

    degree = classify_gray_hist(imgreal, imgshotcut)
    print(degree)
    # cv2.waitKey(0)