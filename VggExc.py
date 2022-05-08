import numpy as np
from PIL import Image
from scipy.spatial.distance import pdist


def euclidean(image1, image2):
    '''欧氏距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'euclidean')[0]


def manhattan(image1, image2):
    '''曼哈顿距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'cityblock')[0]


def chebyshev(image1, image2):
    '''切比雪夫距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'chebyshev')[0]


def cosine(image1, image2):
    '''余弦距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'cosine')[0]


def pearson(image1, image2):
    '''皮尔逊相关系数'''
    X = np.vstack([image1, image2])
    return np.corrcoef(X)[0][1]


def hamming(image1, image2):
    '''汉明距离'''
    return np.shape(np.nonzero(image1 - image2)[0])[0]


def jaccard(image1, image2):
    '''杰卡德距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'jaccard')[0]


def braycurtis(image1, image2):
    '''布雷柯蒂斯距离'''
    X = np.vstack([image1, image2])
    return pdist(X, 'braycurtis')[0]


def mahalanobis(image1, image2):
    '''马氏距离'''
    X = np.vstack([image1, image2])
    XT = X.T
    return pdist(XT, 'mahalanobis')


def jensenshannon(image1, image2):
    '''JS散度'''
    X = np.vstack([image1, image2])
    return pdist(X, 'jensenshannon')[0]


def image_match(image1, image2):
    '''image-match匹配库'''
    try:
        from image_match.goldberg import ImageSignature
    except:
        return -1
    image1 = ImageSignature().generate_signature(image1)
    image2 = ImageSignature().generate_signature(image2)
    return ImageSignature.normalized_distance(image1, image2)


def vgg_match(image1, image2):
    '''VGG16特征匹配'''
    try:
        from numpy import linalg as LA
        from keras.preprocessing import image
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input
    except:
        return -1

    input_shape = (224, 224, 3)
    model = VGG16(weights='imagenet', pooling='max', include_top=False, input_shape=input_shape)

    def extract_feat(img_path):
        '''提取图像特征'''
        img = image.load_img(img_path, target_size=(input_shape[0], input_shape[1]))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        feat = model.predict(img)
        norm_feat = feat[0] / LA.norm(feat[0])
        return norm_feat

    image1 = extract_feat(image1)
    image2 = extract_feat(image2)
    return np.dot(image1, image2.T)


if __name__ == '__main__':
    # 初始化
    image1_name = 'I:/1.png'
    image2_name = 'I:/2.png'
    image3_name = 'I:/3.jpg'

    # 图像预处理
    image1 = Image.open(image1_name).convert('L')  # 转灰度图，若考虑颜色则去掉
    image2 = Image.open(image2_name).convert('L')
    image3 = Image.open(image3_name).convert('L')
    image2 = image2.resize(image1.size)
    image3 = image3.resize(image1.size)
    image1 = np.asarray(image1).flatten()
    image2 = np.asarray(image2).flatten()
    image3 = np.asarray(image3).flatten()

    # 相似度匹配
    print('欧氏距离', euclidean(image1, image2), euclidean(image1, image3))
    print('曼哈顿距离', manhattan(image1, image2), manhattan(image1, image3))
    print('切比雪夫距离', chebyshev(image1, image2), chebyshev(image1, image3))
    print('余弦距离', cosine(image1, image2), cosine(image1, image3))
    print('皮尔逊相关系数', pearson(image1, image2), pearson(image1, image3))
    print('汉明距离', hamming(image1, image2), hamming(image1, image3))
    print('杰卡德距离', jaccard(image1, image2), jaccard(image1, image3))
    print('布雷柯蒂斯距离', braycurtis(image1, image2), braycurtis(image1, image3))
    # print('马氏距离', mahalanobis(image1, image2), mahalanobis(image1, image3))
    print('JS散度', jensenshannon(image1, image2), jensenshannon(image1, image3))

    print('image-match匹配库', image_match(image1_name, image2_name), image_match(image1_name, image3_name))
    print('VGG16特征匹配', vgg_match(image1_name, image2_name), vgg_match(image1_name, image3_name))
