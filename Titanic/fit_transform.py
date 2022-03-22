import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
# # 创建数组
# le = preprocessing.LabelEncoder()
# data_rn = np.random.randint(10,100,10).reshape(5, 2)
# print(data_rn)
# # a=[]
# # a=le.fit(data_rn)
# print(data_rn)
# # print(a)
# # 进行标准归一化
# scaler_mmc = MinMaxScaler()
# scaler_mmc_fit = scaler_mmc.fit(data_rn)
# print(scaler_mmc_fit.data_min_)  # 最小值
# print(scaler_mmc_fit.data_max_)  # 最大值
# print(scaler_mmc_fit.data_range_)  # 极差

#
# from sklearn import preprocessing
# le = preprocessing.LabelEncoder()
# a=le.fit_transform([1,2,2,6,3])
# print(a)
a = np.array([[1,2,3],[4,5,6],[7,8,9]])

b = a.tolist()
print(a)
print(b)
print(len(b))
