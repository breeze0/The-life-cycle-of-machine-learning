from feature_selector import FeatureSelector
import pandas as pd
from sklearn.datasets import load_breast_cancer

breast_cancer = load_breast_cancer()

# print(breast_cancer.data)
data = pd.DataFrame(breast_cancer.data)
data.columns = ["c1","c2","c3","c4","c5","c6","c7","c8","c9","c10",
                "c11","c12","c13","c14","c15","c16","c17","c18","c19","c20",
                "c21","c22","c23","c24","c25","c26","c27","c28","c29","c30"]
target = breast_cancer.target
# air_quality = pd.read_excel('AirQualityUCI.xlsx')
# air_quality['Date'] = pd.to_datetime(air_quality['Date'])
# air_quality['Date'] = (air_quality['Date'] - air_quality['Date'].min()).dt.total_seconds()
# air_quality['Time'] = [int(x.strftime("%H:%M:%S")[:2]) for x in air_quality['Time']]
# air_quality.head()
# labels = air_quality['PT08.S5(O3)']
# air_quality = air_quality.drop(columns = 'PT08.S5(O3)')

# data = pd.read_table('dating1.txt',header=None,encoding='gb2312',delim_whitespace=True)
# data.columns = ["c1","c2","c3","class"]
# labels = data['class']
# data = data.drop(columns = 'class')


fs = FeatureSelector(data = data, labels = target)
#特征缺失值达到阀值
# fs.identify_missing(missing_threshold=0.7)

#只有单个唯一值的特征
# fs.identify_single_unique()
#
#根据阀值选出共线性的特征
fs.identify_collinear(correlation_threshold=0.8)
# fs.plot_collinear()
# list of collinear features to remove
collinear_features = fs.ops['collinear']
# dataframe of collinear features
print(fs.record_collinear.head())
print(collinear_features)
#
fs.identify_zero_importance(task = 'classification',
                            n_iterations = 10, early_stopping = False)
#
# fs.identify_low_importance(cumulative_importance = 0.99)
# # fs.plot_feature_importances(threshold = 0.99, plot_n = 12)

# fs.identify_all(selection_params = {'missing_threshold': 0.5, 'correlation_threshold': 0.7,
#                                     'task': 'regression', 'eval_metric': 'l2',
#                                      'cumulative_importance': 0.9})
# print(fs.ops)