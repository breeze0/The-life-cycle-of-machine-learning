import pandas as pd
from hdfs.client import Client
dataSet=[]
# client = Client("http://lee:50070")#创建连接hdfs的对象
# with client.read("/user/upload/test.txt") as fs:#读取hdfs上的数据
#     f = fs.read()
#     fileds=f.decode('utf8').strip("\n").split("\n")
#     for line in fileds:
#         d = line.split(",")
#         dataSet.append(d)
f = open('test.txt', 'r')
lines = f.readlines()
for line in lines:
    tmp = line.strip("\n").split(",")
    dataSet.append(tmp)

print('原始数据',dataSet)
#转化为DataFrame格式
data = pd.DataFrame(dataSet,columns = ['col1', 'col2', 'col3'])
print (data)
#对一列或多列去重,keep='first'保留第一次出现的,inplace=True在原DataFrame上删除重复
data.drop_duplicates(subset=['col1'], keep='first', inplace=True)
print(data)
#将DataFrame转化为list
data_list = data.values.tolist()
#将列名也一起转化
#data_list = [data.columns.values.tolist()] + data.values.tolist()
print(data_list)