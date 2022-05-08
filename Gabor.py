import cv2
import numpy as np


def gabor_kernel(ksize, sigma, gamma, lamda, alpha, psi):
    '''
    reference
      https://en.wikipedia.org/wiki/Gabor_filter
    '''

    sigma_x = sigma
    sigma_y = sigma / gamma

    ymax = xmax = ksize // 2  # 9//2
    xmin, ymin = -xmax, -ymax
    # print("xmin, ymin,xmin, ymin",xmin, ymin,ymax ,xmax)
    # X(第一个参数，横轴)的每一列一样，  Y（第二个参数，纵轴）的每一行都一样
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))  # 生成网格点坐标矩阵
    # print("y\n",y)
    # print("x\n",x)

    x_alpha = x * np.cos(alpha) + y * np.sin(alpha)
    y_alpha = -x * np.sin(alpha) + y * np.cos(alpha)
    print("x_alpha[0][0]", x_alpha[0][0], y_alpha[0][0])
    exponent = np.exp(-.5 * (x_alpha ** 2 / sigma_x ** 2 +
                             y_alpha ** 2 / sigma_y ** 2))
    # print(exponent[0][0])
    # print(x[0],y[0])
    kernel = exponent * np.cos(2 * np.pi / lamda * x_alpha + psi)
    print(kernel)
    # print(kernel[0][0])
    return kernel


def gabor_filter(gray_img, ksize, sigma, gamma, lamda, psi):
    filters = []
    for alpha in np.arange(0, np.pi, np.pi / 4):
        print("alpha", alpha)
        kern = gabor_kernel(ksize=ksize, sigma=sigma, gamma=gamma,
                            lamda=lamda, alpha=alpha, psi=psi)
        filters.append(kern)

    gabor_img = np.zeros(gray_img.shape, dtype=np.uint8)

    i = 0
    for kern in filters:
        fimg = cv2.filter2D(gray_img, ddepth=cv2.CV_8U, kernel=kern)
        gabor_img = cv2.max(gabor_img, fimg)
        cv2.imwrite("2." + str(i) + "gabor.jpg", gabor_img)
        i += 1
    p = 1.25
    gabor_img = (gabor_img - np.min(gabor_img, axis=None)) ** p
    _max = np.max(gabor_img, axis=None)
    gabor_img = gabor_img / _max
    print(gabor_img)
    gabor_img = gabor_img * 255
    return gabor_img.astype(dtype=np.uint8)


def main():
    src = cv2.imread("I:/DL/HOG/real.jpg")
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gabor_img = gabor_filter(src_gray,
                             ksize=9,
                             sigma=1,
                             gamma=0.5,
                             lamda=5,
                             psi=-np.pi / 2)
    cv2.imwrite("gabor.jpg", gabor_img)
    cv2.imshow('gabor', gabor_img)
    cv2.waitKey(0)  # 等待用户操作

    src = cv2.imread("I:/DL/HOG/shotcut.jpg")
    src_gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    gabor_img = gabor_filter(src_gray,
                             ksize=9,
                             sigma=1,
                             gamma=0.5,
                             lamda=5,
                             psi=-np.pi / 2)
    cv2.imwrite("gabor.jpg", gabor_img)
    cv2.imshow('gabor', gabor_img)
    cv2.waitKey(0)  # 等待用户操作


main()