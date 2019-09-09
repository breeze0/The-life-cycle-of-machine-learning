from sklearn import datasets
import matplotlib.pyplot as plt

# digits = datasets.load_digits()
# print(digits.target[2])
# print(digits.images[2])
# plt.figure()
# plt.axis('off')
# plt.imshow(digits.images[2],cmap=plt.cm.gray_r,interpolation='nearest')
# plt.show()

import numpy as np
from skimage.feature import corner_harris, corner_peaks
from skimage.color import rgb2gray
import skimage.io as io
from skimage.exposure import equalize_hist

#提取感兴趣点
def show_corners(corners, image):
    fig = plt.figure()
    plt.gray()
    plt.imshow(image)
    y_corner, x_corner = zip(*corners)
    plt.plot(x_corner, y_corner, 'or')
    plt.xlim(0, image.shape[1])
    plt.ylim(image.shape[0], 0)
    fig.set_size_inches(np.array(fig.get_size_inches()) * 1.5)
    plt.show()

# mandrill = io.imread('husky.jpg')
# #归一化
# mandrill = equalize_hist(rgb2gray(mandrill))
# corners = corner_peaks(corner_harris(mandrill), min_distance=2)
# show_corners(corners, mandrill)

#surf
import mahotas as mh
from mahotas.features import surf

image = mh.imread('husky.jpg', as_grey=True)
print('第一个surf描述符: \n{}\n'.format(surf.surf(image)[0]))
print('抽取了%s个SURF描述符'%len(surf.surf(image)))