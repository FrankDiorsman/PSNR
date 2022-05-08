# encoding:utf-8
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread('I:/raw1.jpg', cv2.IMREAD_GRAYSCALE)
equ = cv2.equalizeHist(img)

plt.subplot(221), plt.imshow(img, 'gray'), plt.title('img'), plt.xticks([]), plt.yticks([])
plt.subplot(222), plt.imshow(equ, 'gray'), plt.title('equ'), plt.xticks([]), plt.yticks([])
plt.subplot(223), plt.hist(img.ravel(), 256), plt.title('img_hist')
plt.subplot(224), plt.hist(equ.ravel(), 256), plt.title('equ_hist')

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()