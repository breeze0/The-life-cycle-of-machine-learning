from sklearn import datasets
import matplotlib.pyplot as plt

digits = datasets.load_digits()
print(digits.target[2])
print(digits.images[2])
plt.figure()
plt.axis('off')
plt.imshow(digits.images[2],cmap=plt.cm.gray_r,interpolation='nearest')
plt.show()