from feature_selector import FeatureSelector
import pandas as pd

air_quality = pd.read_excel('AirQualityUCI.xlsx')
air_quality['Date'] = pd.to_datetime(air_quality['Date'])
air_quality['Date'] = (air_quality['Date'] - air_quality['Date'].min()).dt.total_seconds()
air_quality['Time'] = [int(x.strftime("%H:%M:%S")[:2]) for x in air_quality['Time']]
air_quality.head()

labels = air_quality['PT08.S5(O3)']
air_quality = air_quality.drop(columns = 'PT08.S5(O3)')

fs = FeatureSelector(data = air_quality, labels = labels)
# #特征缺失值达到阀值
# fs.identify_missing(missing_threshold=0.6)
#
# #只有单个唯一值的特征
# fs.identify_single_unique()
#
# #根据阀值选出共线性的特征
fs.identify_collinear(correlation_threshold=0.7)
fs.plot_collinear()
# # list of collinear features to remove
collinear_features = fs.ops['collinear']
# # dataframe of collinear features
print(fs.record_collinear.head())
# #print(collinear_features)

fs.identify_zero_importance(task = 'regression', eval_metric = 'auc',
                            n_iterations = 10, early_stopping = True)

fs.identify_low_importance(cumulative_importance = 0.99)
fs.plot_feature_importances(threshold = 0.99, plot_n = 12)

print(fs.ops)