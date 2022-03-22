import numpy as np

import pandas as pd
pd.set_option('display.float_format', lambda x: '{:.2f}'.format(x))

import seaborn as sns
color = sns.color_palette()
sns.set_style('darkgrid')

import matplotlib.pyplot as plt


from scipy import stats
from scipy.special import boxcox1p
from scipy.stats import norm, skew

#忽略警告
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn

from sklearn.preprocessing import LabelEncoder

train = pd.read_csv('train.csv')
print('The shape of training data:', train.shape)
train.head()
test = pd.read_csv('test.csv')

#ID列没有用，直接删掉
train.drop('Id', axis=1, inplace=True)
test.drop('Id', axis=1, inplace=True)
# print('The shape of training data:', train.shape)
# print('The shape of testing data:', test.shape)

#绘制目标值分布
sns.distplot(train['SalePrice'])
plt.show()

#pandas数据介绍（发现最大值与均值差距较大，可能有异常值）
# print(train['SalePrice'].describe())

#分离数字特征和类别特征
num_features = []
cate_features = []
for col in test.columns:#遍历所有的列标签
    if test[col].dtype == 'object':
        cate_features.append(col)
    else:
        num_features.append(col)
print('number of numeric features:', len(num_features))
print('number of categorical features:', len(cate_features))


# #查看数字特征与目标值的关系
# plt.figure(figsize=(16, 20))
# plt.subplots_adjust(hspace=0.3, wspace=0.3)#保留宽度与高度
# for i, feature in enumerate(num_features):
#     plt.subplot(9, 4, i+1)
#     sns.scatterplot(x=feature, y='SalePrice', data=train, alpha=0.5)
#     plt.xlabel(feature)
#     plt.ylabel('SalePrice')
# plt.show()#‘TotalBsmtSF’、'GrLiveArea’与目标值之间有明显的线性关系

#查看‘Neighborhood’与目标值的关系
# plt.figure(figsize=(16, 12))
# sns.boxplot(x='Neighborhood', y='SalePrice', data=train)
# plt.xlabel('Neighborhood', fontsize=14)
# plt.ylabel('SalePrice', fontsize=14)
# plt.xticks(rotation=90, fontsize=12)

corrs = train.corr()
plt.figure(figsize=(16, 16))
sns.heatmap(corrs)


#分析与目标值相关度最高的十个变量
cols_10 = corrs.nlargest(10, 'SalePrice')['SalePrice'].index
corrs_10 = train[cols_10].corr()
plt.figure(figsize=(6, 6))
sns.heatmap(corrs_10, annot=True)

g = sns.PairGrid(train[cols_10])
g.map_diag(plt.hist)
g.map_offdiag(plt.scatter)
plt.show()

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)
#处理掉右下的明显异常值
train = train.drop(train[(train['TotalBsmtSF']>6000) & (train['SalePrice']<200000)].index)

sns.scatterplot(x='TotalBsmtSF', y='SalePrice', data=train)

#对’GrLiveArea’进行同样的处理：
sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
#处理掉右下的异常值
train = train.drop(train[(train['GrLivArea']>4000) & (train['SalePrice']<200000)].index)

sns.scatterplot(x='GrLivArea', y='SalePrice', data=train)
#查看训练集中各特征的数据缺失个数
print('The shape of training data:', train.shape)
train_missing = train.isnull().sum()
train_missing = train_missing.drop(train_missing[train_missing==0].index).sort_values(ascending=False)
print(train_missing)

#说明文档中没有的即填充none
none_lists = ['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtFinType1',
              'BsmtFinType2', 'BsmtCond', 'BsmtExposure', 'BsmtQual', 'MasVnrType']
for col in none_lists:
    train[col] = train[col].fillna('None')
    test[col] = test[col].fillna('None')

#说明文档里丢失的数据补为频率最高的值
most_lists = ['MSZoning', 'Exterior1st', 'Exterior2nd', 'SaleType', 'KitchenQual', 'Electrical']
for col in most_lists:
    train[col] = train[col].fillna(train[col].mode()[0])  #mode()得到的结果是一个pd.Series，[0]是取其第一个元素，即出现次数最多的那个元素
    test[col] = test[col].fillna(train[col].mode()[0])    #注意这里补充的是训练集中出现最多的类别

#Functional中补充Typ，Utilities无用处直接删去
train['Functional'] = train['Functional'].fillna('Typ')
test['Functional'] = test['Functional'].fillna('Typ')

train.drop('Utilities', axis=1, inplace=True)
test.drop('Utilities', axis=1, inplace=True)

#一些特征值补零
zero_lists = ['GarageYrBlt', 'MasVnrArea', 'BsmtFullBath', 'BsmtHalfBath', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'GarageCars', 'GarageArea',
              'TotalBsmtSF']
for col in zero_lists:
    train[col] = train[col].fillna(0)
    test[col] = test[col].fillna(0)

#补充为Neighborhood的平均数
train['LotFrontage'] = train.groupby('LotConfig')['LotFrontage'].apply(lambda x: x.fillna(x.mean()))#以Neighborhood对LotFrontage进行聚合并在最后一行加上平均数
for ind in test['LotFrontage'][test['LotFrontage'].isnull().values==True].index:
    x = test['LotConfig'].iloc[ind]
    test['LotFrontage'].iloc[ind] = train.groupby('LotConfig')['LotFrontage'].mean()[x]

print(train.isnull().sum().any())#.sum统计空列表的个数

#从存放类别特征的列表去掉'Utilities'
cate_features.remove('Utilities')
print('The number of categorical features:', len(cate_features))

#对不能输入模型的类别特征进行编码，对于各个类别中可能存在顺序关系的，用LabelEncoder编码，对于不存在顺序关系的，用get_dummies进行编码
for col in cate_features:
    train[col] = train[col].astype(str)
    test[col] = test[col].astype(str)
le_features = ['Street', 'Alley', 'LotShape', 'LandContour', 'LandSlope', 'HouseStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'ExterQual',
               'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir',
               'KitchenQual', 'Functional', 'FireplaceQu', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC', 'Fence']
for col in le_features:
    encoder = LabelEncoder()
    value_train = set(train[col].unique())
    value_test = set(test[col].unique())
    value_list = list(value_train | value_test)
    encoder.fit(value_list)
    train[col] = encoder.transform(train[col])
    test[col] = encoder.transform(test[col])

#处理偏斜度大于0.5的数据
skewness = train[num_features].apply(lambda x: skew(x)).sort_values(ascending=False)
skewness = skewness[skewness>0.5]
skew_features = skewness.index

#增强模型对异常度的刚性，用Box Cox转换来处理偏斜数据
for col in skew_features:
    lam = stats.boxcox_normmax(train[col]+1)    #+1是为了保证输入大于零
    train[col] = boxcox1p(train[col], lam)
    test[col] = boxcox1p(test[col], lam)

#依据经验构建的新的特征
train['IsRemod'] = 1
train['IsRemod'].loc[train['YearBuilt']==train['YearRemodAdd']] = 0  #是否翻新(翻新：1， 未翻新：0)
train['BltRemodDiff'] = train['YearRemodAdd'] - train['YearBuilt']  #翻新与建造的时间差（年）
train['BsmtUnfRatio'] = 0
train['BsmtUnfRatio'].loc[train['TotalBsmtSF']!=0] = train['BsmtUnfSF'] / train['TotalBsmtSF']  #Basement未完成占总面积的比例
train['TotalSF'] = train['TotalBsmtSF'] + train['1stFlrSF'] + train['2ndFlrSF']  #总面积
#对测试集做同样的处理
test['IsRemod'] = 1
test['IsRemod'].loc[test['YearBuilt']==test['YearRemodAdd']] = 0  #是否翻新(翻新：1， 未翻新：0)
test['BltRemodDiff'] = test['YearRemodAdd'] - test['YearBuilt']  #翻新与建造的时间差（年）
test['BsmtUnfRatio'] = 0
test['BsmtUnfRatio'].loc[test['TotalBsmtSF']!=0] = test['BsmtUnfSF'] / test['TotalBsmtSF']  #Basement未完成占总面积的比例
test['TotalSF'] = test['TotalBsmtSF'] + test['1stFlrSF'] + test['2ndFlrSF']  #总面积

#集中处理数据
dummy_features = list(set(cate_features).difference(set(le_features)))
all_data = pd.concat((train.drop('SalePrice', axis=1), test)).reset_index(drop=True)
all_data = pd.get_dummies(all_data, drop_first=True)  #注意独热编码生成的时候要去掉一个维度，保证剩下的变量都是相互独立的

#保存最终处理结果
trainset = all_data[:1458]
y = train['SalePrice']
trainset['SalePrice'] = y.values
testset = all_data[1458:]
print('The shape of training data:', trainset.shape)
print('The shape of testing data:', testset.shape)

trainset.to_csv('train_data.csv', index=False)
testset.to_csv('test_data.csv', index=False)





