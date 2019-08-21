import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
from sklearn.ensemble import RandomForestRegressor
from hdfs.client import Client
#读取数据
dataSet=[]
# client = Client("http://lee:50070")#创建连接hdfs的对象
# with client.read("/user/upload/missingdata.csv") as fs:#读取hdfs上的数据
#     f = fs.read()
#     fileds=f.decode('utf8').strip("\r\n").split("\r\n")
#     for line in fileds:
#         d = line.split(",")
#         dataSet.append(d)
# print('原始数据',dataSet)
# data = pd.DataFrame(dataSet,columns = ['a','b','c','d','e'])
data = pd.read_csv('missingdata.csv', encoding='UTF8')
data.columns = ['a','b','c','d','e']

#1.直接删除
#数据缺失值较少，可以直接删除。注意，在计算缺失值时，对于缺失值不是NaN的要用replace()函数替换成NaN格式，否则pd.isnull()检测不出来
#将空值形式的缺失值转换成可以识别的类型
data = data.replace(' ',np.NaN)
print (data.columns)
#将每列中缺失值的个数统计出来
null_all = data.isnull().sum()
print (null_all)
#查看第a列有缺失值的数据missingvalue.py:23
a_null = data[pd.isnull(data['a'])]
print (a_null)
#a列缺失占比
a_ratio = len(data[pd.isnull(data['a'])])/len(data)
print (a_ratio)
#丢弃缺失值，将存在缺失值的行丢失
# new_drop = data.dropna(axis = 0)
# print (new_drop)
#丢弃某几列有缺失值的行
# new_drop2 = data.dropna(axis = 0,subset=['a','b'])
# print (new_drop2)



#2.使用一个全局常量填充缺失值
#可以用一个常数来填充缺失值
#用0填充缺失值
# fill_data = data.fillna('Unknow')
# print (fill_data.isnull().sum())
# print(fill_data)




#3.均值、众数、中位数填充。均值即为'mean'，众数为'median'
imr = Imputer(missing_values = 'NaN',strategy = 'mean',axis = 0)
imr = imr.fit(data.values)
imputed_data = pd.DataFrame(imr.transform(data.values))
print (imputed_data)


#4.插值法
#插值法，计算的是缺失值前一个值和后一个值的平均数
#data['a'] = data['a'].interpolate()












