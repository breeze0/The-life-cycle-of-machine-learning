from sklearn.neighbors import NearestNeighbors
import numpy as np

class ENN:
    """
    ENN欠采样算法.(Edited Nearest Neighbours)

    对于属于多数类的样本，如果k个近邻点有超过一半不属于多数类，则这个样本被剔除

    Parameters:
    -----------
    k: int
        选取的近邻数目.
    """
    def __init__(self,k=5):
        self.k = k

    def fit(self, X, y):
        negative_X = X[y == 0]
        postive_X = X[y == 1]
        negative_y = y[y == 0]
        postive_y = y[y == 1]
        #要被剔除的数据
        selected = []
        knn = NearestNeighbors(n_neighbors=self.k).fit(X)
        for i in range(len(negative_X)):
            # 多数类的样本在全样本中的k近邻
            k_neighbors = knn.kneighbors(negative_X[i].reshape(1, -1), return_distance=False)[0]
            # 计算近邻中少数类的比例
            rate = self.evaluate(k_neighbors, y)
            if rate < 0.2:
                selected.append(i)

        x_list = negative_X[selected]
        y_list = negative_y[selected]

        return (np.concatenate((postive_X, x_list), axis=0),
                np.concatenate((postive_y, y_list), axis=0))

    def evaluate(self, x, y):
        count = 0
        for v in x:
            if y[v] == 1:
                count = count+1

        return count / float(len(x))

X = np.array([[1, 3, 2], [1, 2, 3], [3, 4, 6], [2, 2, 1], [3, 5, 2], [5, 3, 4], [3, 2, 4], [5, 4, 3], [6, 5, 2], [4, 5, 6]])
y = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
enn = ENN(k=5)
print(enn.fit(X, y))