from sklearn.preprocessing import OneHotEncoder
import pandas as pd

#原始数据集
data = [['大众',4,130000],
        ['Toyota',5,150000],
        ['奥迪',2,250000],
        ['宝马',18,360000],
        ['奔驰',7,490000],
        ['凯迪拉克',30,520000],
        ['英菲尼迪',2,364000]]

data = pd.DataFrame(data, columns = ['carmodel', 'drivingage', 'carprice'])
print (data)

#把带中文的标称属性转换为数值型
listUniq = data.ix[:, 'carmodel'].unique()
for i in range(len(listUniq)):
    data.ix[:,'carmodel'] = data.ix[:, 'carmodel'].apply(lambda x:i if x == listUniq[i] else x)
print (data)

#进行独热编码
tempdata = data[['carmodel']]
print (tempdata)
enc = OneHotEncoder()
enc.fit(tempdata)

#独热编码的结果转换成二位数组
tempdata = enc.transform(tempdata).toarray()
print (tempdata)
#print ('取值范围的整数个数：',enc.n_values_)

#再将二位数组转换成DataFrame
tempdata = pd.DataFrame(tempdata,columns = ['carmodel']*len(tempdata[0]))
print (tempdata)