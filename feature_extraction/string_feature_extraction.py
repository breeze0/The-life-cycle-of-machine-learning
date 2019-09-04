from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

#分类变量特征提取
#不连续的变量，通常用one-hot编码
onehot_encoder = DictVectorizer()
instance = [{'city':'BeiJing'},{'city':'ShangHai'},{'city':'ChengDu'}]
print(onehot_encoder.fit_transform(instance).toarray())

#文字特征提取
#词库模型表示法
#构建所有文档中出现过的单词的文集，文集中有多少个单词，每个文档就由多少维向量构成
corpus = ['UNC played Duke in basketball',
          'Duke lost the basketball game',
          'I ate a sanwich']
vectorizer = CountVectorizer(stop_words='english')
print(vectorizer.fit_transform(corpus).todense())
print(vectorizer.vocabulary_)
