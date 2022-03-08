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
    train[col] = train[col].fillna(train[col].mode()[0])
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

#补充为Neighborhood的中位数
train['LotFrontage'] = train.groupby('Neighborhood')['LotFrontage'].apply(lambda x: x.fillna(x.median()))
for ind in test['LotFrontage'][test['LotFrontage'].isnull().values==True].index:
    x = test['Neighborhood'].iloc[ind]
    test['LotFrontage'].iloc[ind] = train.groupby('Neighborhood')['LotFrontage'].median()[x]

