import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 算法思路
# 设有n条d维数据
# 1.将原始数据按列组成 n 行 d 列矩阵 X
# 2.将 X 的每一列（代表一个属性）进行零均值化，即减去这一列的均值
# 3.求出协方差矩阵 C = (1/m)*X*XT
# 4.求出协方差矩阵的特征值及对应的特征向量
# 5.将特征向量按对应特征值大小从上到下按行排列成矩阵，取前k行组成矩阵 P
# 6.Y=PX 即为降维到 K 维后的数据

def loadDataSet(filename):
    df = pd.read_table(filename, sep='\t')
    return np.array(df)

def showData(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0], dataMat[:, 1], c='green')
    ax.scatter(np.array(reconMat[:, 0]), reconMat[:, 1], c='red')
    plt.show()

def pca(dataMat, topNfeat=999999):

    # 1.对所有样本进行中心化（所有样本属性减去属性的平均值）
    meanVals = np.mean(dataMat, axis=0)
    meanRemoved = dataMat - meanVals

    # 2.计算样本的协方差矩阵 XXT
    covmat = np.cov(meanRemoved, rowvar=0)
    print(covmat)

    # 3.对协方差矩阵做特征值分解，求得其特征值和特征向量，并将特征值从大到小排序，筛选出前topNfeat个
    eigVals, eigVects = np.linalg.eig(np.mat(covmat))
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]    # 取前topNfeat大的特征值的索引
    redEigVects = eigVects[:, eigValInd]        # 取前topNfeat大的特征值所对应的特征向量

    # 4.将数据转换到新的低维空间中
    lowDDataMat = meanRemoved * redEigVects     # 降维之后的数据
    reconMat = (lowDDataMat * redEigVects.T) + meanVals # 重构数据，可在原数据维度下进行对比查看
    return np.array(lowDDataMat), np.array(reconMat)

air_quality = pd.read_excel('AirQualityUCI.xlsx')
air_quality['Date'] = pd.to_datetime(air_quality['Date'])
air_quality['Date'] = (air_quality['Date'] - air_quality['Date'].min()).dt.total_seconds()
air_quality['Time'] = [int(x.strftime("%H:%M:%S")[:2]) for x in air_quality['Time']]
dataMat = np.array(air_quality)
print(dataMat.shape)
lowDDataMat, reconMat = pca(dataMat, 5)
showData(dataMat, reconMat)
print(lowDDataMat.shape)