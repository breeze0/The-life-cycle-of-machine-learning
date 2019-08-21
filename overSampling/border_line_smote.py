import random
from sklearn.neighbors import NearestNeighbors
import numpy as np
import smote

class BoderLineSmote:
    def __init__(self, k1=5, k2=3, sampling_rate=5):
        self.sampling_rate = sampling_rate
        self.k1 = k1
        self.k2 = k2
        self.newindex = 0

    def classify(self, X, y):
        negative_X = X[y == 0]
        postive_X = X[y == 1]
        noise = []
        danger = []
        safe = []
        knn = NearestNeighbors(n_neighbors=self.k1).fit(X)

        for i in range(len(postive_X)):
            # reshape(1,-1)重组数组,-1表示列自动计算
            k_neighbors = knn.kneighbors(X[i].reshape(1, -1), return_distance=False)[0]
            # 对正样本集(minority class samples)中每个样本, 分别根据所有样本集生成k个近邻
            rate = self.evaluate(k_neighbors, y)
            # 计算近邻中正样本所占的比例
            if rate == 0:
                noise.append(i)
            elif rate < 0.5:
                danger.append(i)
            else:
                safe.append(i)

        return danger

    def evaluate(self, x, y):
        count = 0
        for v in x:
            if y[v] == 1:
                count = count+1

        return count / float(len(x))

    def fit(self, X, y=None):
        origin_X = X
        danger = self.classify(origin_X, y)
        print(danger)

        negative_X = X[y == 0]
        X = X[y == 1]

        # 初始化一个矩阵, 用来存储合成样本
        self.synthetic = []

        # 找出正样本集(数据集X)中的每一个样本在数据集X中的k个近邻
        knn = NearestNeighbors(n_neighbors=self.k2).fit(X)
        for i in danger:
            k_neighbors = knn.kneighbors(origin_X[i].reshape(1, -1), return_distance=False)[0]
            # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成
            # sampling_rate个新的样本
            self.synthetic_samples(origin_X, X, i, k_neighbors)

        return (np.concatenate((self.synthetic, X, negative_X), axis=0),
                np.concatenate(([1] * (len(self.synthetic) + len(X)), y[y == 0]), axis=0))


    # 对正样本集(minority class samples)中每个样本, 分别根据其k个近邻生成sampling_rate个新的样本
    def synthetic_samples(self, origin_X, X, i, k_neighbors):
        for j in range(self.sampling_rate):
            # 从k个近邻里面随机选择一个近邻
            neighbor = np.random.choice(k_neighbors)
            # 计算样本X[i]与刚刚选择的近邻的差
            diff = X[neighbor] - origin_X[i]
            # 生成新的数据
            self.synthetic.append(origin_X[i] + random.random() * diff)





X = np.array([[1, 3, 2], [1, 2, 3], [3, 4, 6], [2, 2, 1], [3, 5, 2], [5, 3, 4], [3, 2, 4], [5, 4, 3], [6, 5, 2], [4, 5, 6]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
bs = BoderLineSmote(k1=5, k2=3, sampling_rate=1)
print(bs.fit(X, y))
