import cv2
import numpy as np

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

def rgb2hsv(img):
    h = img.shape[0]
    w = img.shape[1]
    H = np.zeros((h,w),np.float32)
    S = np.zeros((h, w), np.float32)
    V = np.zeros((h, w), np.float32)
    r,g,b = cv2.split(img)
    r, g, b = r/255.0, g/255.0, b/255.0
    for i in range(0, h):
        for j in range(0, w):
            mx = max((b[i, j], g[i, j], r[i, j]))
            mn = min((b[i, j], g[i, j], r[i, j]))
            V[i, j] = mx
            if V[i, j] == 0:
                S[i, j] = 0
            else:
                S[i, j] = (V[i, j] - mn) / V[i, j]
            if mx == mn:
                H[i, j] = 0
            elif V[i, j] == r[i, j]:
                if g[i, j] >= b[i, j]:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn))
                else:
                    H[i, j] = (60 * ((g[i, j]) - b[i, j]) / (V[i, j] - mn))+360
            elif V[i, j] == g[i, j]:
                H[i, j] = 60 * ((b[i, j]) - r[i, j]) / (V[i, j] - mn) + 120
            elif V[i, j] == b[i, j]:
                H[i, j] = 60 * ((r[i, j]) - g[i, j]) / (V[i, j] - mn) + 240
            H[i,j] = H[i,j] / 2
    return H, S, V

def hist_compare(image1, image2):
    hist1 = create_HSV_hist(image1)
    hist2 = create_HSV_hist(image2)
    match1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
    match2 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    match3 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)
    print("巴氏距离: %s, 相关性: %s, 卡方: %s" % (match1, match2, match3))

img = cv2.imread('I:/DL/HOG/real.jpg')
img2 = cv2.imread('I:/DL/HOG/shotcut.jpg')
h,s,v = rgb2hsv(img)
hsvHist = create_HSV_hist(img)
cv2.imshow("h",h)
cv2.imshow("s",s)
cv2.imshow("v",v)
height = img.shape[0]
width = img.shape[1]
h1 = h/(width*width)
s1 = s/(width*width)
v1 = v/(width*width)
print(h1,s1,v1)

hist_compare(img,img2)

merged = cv2.merge([h,s,v]) #前面分离出来的三个通道
cv2.imshow("hsv",merged)
cv2.waitKey(0)
cv2.destroyAllWindows()
