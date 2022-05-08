import cv2
import matplotlib.pyplot as plt

# 计算方差
def getss(list):
    # 计算平均值
    avg = sum(list) / len(list)
    # 定义方差变量ss，初值为0
    ss = 0
    # 计算方差
    for l in list:
        ss += (l - avg) * (l - avg) / len(list)
    # 返回方差
    return ss


# 获取每行像素平均值
def getdiff(img):
    # 定义边长
    Sidelength = 4096
    # 缩放图像
    img = cv2.resize(img, (Sidelength, Sidelength), interpolation=cv2.INTER_CUBIC)
    # 灰度处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 标准化处理
    cv2.normalize(gray, gray, 0, 1, cv2.NORM_MINMAX)
    # avglist列表保存每行像素平均值
    avglist = []
    # 计算每行均值，保存到avglist列表
    for i in range(Sidelength):
        avg = sum(gray[i]) / len(gray[i])
        avglist.append(avg)
    # 返回avglist平均值
    return avglist


# 读取测试图片
raw = cv2.imread("I:/raw1.jpg")
diffraw = getdiff(raw)
print('raw:', getss(diffraw))

# 读取测试图片
shot = cv2.imread("I:/shot1.png")
diffshot = getdiff(shot)
print('shot:', getss(diffshot))

ss1 = getss(diffraw)
ss2 = getss(diffshot)
print("两张照片的方差为：%s" % (abs(ss1 - ss2)))

x = range(4096)

plt.figure("avg")
plt.plot(x, diffraw, marker="*", label="raw")
plt.plot(x, diffshot, marker="*", label="shot")
plt.title("avg")
plt.legend()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()