import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
#源数据为线性不可分的
X, y = make_moons(n_samples=100, random_state=123)

plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='^', alpha=0.5)
plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', alpha=0.5)

plt.tight_layout()

plt.show()

#核主成分分析（Kernelized PCA，KPCA）利用核技巧将d维线性不可分的输入空间映射到线性可分的高维特征空间D中，
#然后对特征空间进行PCA降维，将维度降到d‘维
#并利用核技巧简化计算。也就是一个先升维后降维的过程，这里的维度满足d’<d<D

# 参考博客 https://blog.csdn.net/weixin_40604987/article/details/79632888

from sklearn.decomposition import KernelPCA

X, y = make_moons(n_samples=100, random_state=123)
print(X.shape)
scikit_kpca = KernelPCA(n_components=2, kernel='rbf', gamma=15)
X_skernpca = scikit_kpca.fit_transform(X)
print(X_skernpca.shape)

plt.scatter(X_skernpca[y == 0, 0], X_skernpca[y == 0, 1],
            color='red', marker='^', alpha=0.5)
plt.scatter(X_skernpca[y == 1, 0], X_skernpca[y == 1, 1],
            color='blue', marker='o', alpha=0.5)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.tight_layout()

plt.show()
