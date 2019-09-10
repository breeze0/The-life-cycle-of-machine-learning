from sklearn.datasets import load_iris
import numpy as np
import matplotlib.pyplot as plt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis



# LDA是通过求得一个变换W,使得变换之后的新均值之差最大、方差最大（也就是最大化类间距离和最小化类内距离），变换W就是特征的投影方向。
# LDA特征提取过程

# 将样本分类
# 求类内散度矩阵Sw
# 求类间散度矩阵Sb
# 计算Sw^(-1)*Sb的特征值和特征矩阵
# 特征值排序，提取前K个特征向量
# 参考博客 https://blog.csdn.net/u012679707/article/details/80551628

def LDA_reduce_dimension(X, y, nComponents):
    '''
    输入：X为数据集(m*n)，y为label(m*1)，nComponents为目标维数
    输出：W 矩阵（n * nComponents）
    '''
    # y1= set(y) #set():剔除矩阵y里的重复元素,化为集合的形式
    labels = list(set(y))  # list():将其转化为列表

    xClasses = {}  # 字典
    for label in labels:
        xClasses[label] = np.array([X[i] for i in range(len(X)) if y[i] == label])  # list解析
    # xClasses字典key为类别, value为属于该类的数据

    # 整体均值
    meanAll = np.mean(X, axis=0)  # 按列求均值，结果为1*n(行向量)
    meanClasses = {}

    # 求各类均值
    for label in labels:
        meanClasses[label] = np.mean(xClasses[label], axis=0)  # 1*n

    # 全局散度矩阵
    St = np.zeros((len(meanAll), len(meanAll)))
    St = np.dot((X - meanAll).T, X - meanAll)

    # 求类内散度矩阵
    # Sw=sum(np.dot((Xi-ui).T, Xi-ui))   i=1...m
    Sw = np.zeros((len(meanAll), len(meanAll)))  # n*n
    for i in labels:
        Sw += np.dot((xClasses[i] - meanClasses[i]).T, (xClasses[i] - meanClasses[i]))

    # 求类间散度矩阵
    Sb = np.zeros((len(meanAll), len(meanAll)))  # n*n
    Sb = St - Sw

    # 求类间散度矩阵
    # Sb=sum(len(Xj) * np.dot((uj-u).T,uj-u))  j=1...k
    # Sb=np.zeros((len(meanAll), len(meanAll) )) # n*n
    # for i in labels:
    #     Sb+= len(xClasses[i]) * np.dot( (meanClasses[i]-meanAll).T.reshape(len(meanAll),1),
    #                                     (meanClasses[i]-meanAll).reshape(1,len(meanAll))
    #                                )

    # 计算Sw-1*Sb的特征值和特征矩阵
    eigenValues, eigenVectors = np.linalg.eig(
        np.dot(np.linalg.inv(Sw), Sb)
    )
    # 提取前nComponents个特征向量
    sortedIndices = np.argsort(eigenValues)  # 特征值排序
    W = eigenVectors[:, sortedIndices[:-nComponents - 1:-1]]  # 提取前nComponents个特征向量
    return W



iris = load_iris()
X = iris.data
print(X.shape)
y = iris.target
# LDA特征提取
W = LDA_reduce_dimension(X, y, 2) #得到投影矩阵
newX = np.dot(X, W)
print(newX.shape)
#绘图
plt.figure()
plt.scatter(newX[:, 0], newX[:, 1], c=y, marker='o')  # 颜色c取y的类数,第一列数据为横轴，第二列数据为纵轴
plt.title('My_LDA')

#sklearn自带库函数
lda_sklearn = LinearDiscriminantAnalysis(n_components=2)
lda_sklearn.fit(X, y)
newX1 = lda_sklearn.transform(X)
plt.figure(2)
plt.scatter(newX1[:, 0], newX1[:, 1], marker='o', c=y)
plt.title('sklearn_LDA')

plt.show()



