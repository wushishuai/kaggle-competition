import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
import math
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

#读入训练集和测试集及其标签：
train_data = pd.read_csv('mytrain.csv', lineterminator='\n')
test_data=pd.read_csv('mytest.csv', lineterminator='\n')
test_Lable=pd.read_csv('mygender.csv', lineterminator='\n')
testLable=list(map(int,test_Lable['Survived']))
print(train_data.columns)
#利用pandas.read_csv可以读入表格数据且.columns可以直接查看每一列的表头。


#看一下训练集和测试集的各特征的缺失情况：
for column in test_data.columns:
    print(column,':',train_data[column].count()/len(train_data),'  ',train_data[column].count())
    print(column,':',test_data[column].count()/len(test_data),'  ',test_data[column].count())
#更直接的做法如下：（取消掉注释可以直接运行）
#train_data.info#可以直接查看每一列的数据个数、类型等情况


#无效特征处理：
#PassengerId肯定是和结果没关系的，删掉
#Ticket票号和Cabin一样情况复杂难以分析，删掉
#Name比较特殊，他其中是有一些有用的信息的，比如Mr和Mrs就蕴含了性别信息，而诸如master之类的尊称又可以反映社会地位（一定程度上和船舱号、消费等有关），因而其实是可以保留的。
#但是以来分析起来比较复杂，二来其携带的性别、社会地位、消费能力等信息可以从Sex、Fare等特征中得到反映，所以这里选择直接删掉。
train_data=train_data.drop(['\r'],axis=1)
test_data=test_data.drop(['\r'],axis=1)
train_data=train_data.drop(['PassengerId'],axis=1)
test_data=test_data.drop(['PassengerId'],axis=1)
train_data=train_data.drop(['Name'],axis=1)
test_data=test_data.drop(['Name'],axis=1)
train_data=train_data.drop(['Cabin'],axis=1)
test_data=test_data.drop(['Cabin'],axis=1)
train_data=train_data.drop(['Ticket'],axis=1)
test_data=test_data.drop(['Ticket'],axis=1)

print(len(train_data))
print(len(test_data))

#训练集有缺失的都是坏数据，删了：
train_data.dropna(inplace=True)
trainLen=len(train_data)
testLen=len(test_data)
print(trainLen,testLen)
#训练集标签特征分离：
trainLable= list(map(int,train_data['Survived']))
train_data=train_data.drop(['Survived'],axis=1)
print(train_data.columns)

#
#处理一下测试集里的缺失值，测试集的缺失数据不能删
#处理Fare，先看一下分布，发现明显有个众数非常突出，且训练集和测试集众数接近：
test_data['Fare']=test_data['Fare'].fillna(test_data['Fare'].dropna().mode()[0])
#此处，test_data['Fare']表示Fare那一列，.fillna后面的内容即规定了将.dropna()空的内容，补为mode()[0]众数，加【0】是为了防止众数有多个，故取第一个。
#把训练集和测试集合起来编码：
combineData=train_data.append(test_data)
print(len(combineData.columns))

#先编码后拆分：
def getReview(data,changeColumns):
    ResultReview=[]
    listReview=data
    le = LabelEncoder()
    for column in changeColumns:
        listData=[]
        for review in data[column]:
            listData.append(review)
            # print(review)
            # print(listData)
        listReview[column]=le.fit_transform(listData)
        #fit与transform实际上是分开的，le为LabelEncoder()le.fit是对listData中的内容进行编码le.transform为按照编码内容对现有的数组进行编码。
        #这一步的实质过程是在data中挑选出changeColumns中的几类然后将其进行编码
        # print(listReview)
    #向量化（需要一个个的append）：
    for i in range(len(data)):
        rowVec=[]
        for j in range(0,len(data.columns)):
            rowVec.append(listReview.iloc[i,j])
        ResultReview.append(rowVec)
    return ResultReview
#
changeColumns=['Sex','Embarked']
combine_Review=np.array(getReview(combineData,changeColumns))

scl = MinMaxScaler()#将数据归一化的函数
combineReview=scl.fit_transform(combine_Review)
trainReview=combineReview[0:trainLen]#将合并好的数据再次按照长度进行划分
testReview=combineReview[trainLen:trainLen+testLen]

#处理Age缺失值：
#获取空元素下标：
isNull=test_data['Age'].isnull().tolist()#.isnull()将空缺数据的部分变为false，.tolist()将转变后的原数组转变为一行的形式
listAgeTrain=[]
listAgeTest=[]
for elem in trainReview:
    listAgeTrain.append(elem)
for i in range(0,len(isNull)):
    if isNull[i]==False:
        listAgeTrain.append(testReview[i])
    else:
        listAgeTest.append(testReview[i])
ageTrain = np.array(listAgeTrain)
ageTest=np.array(listAgeTest)
# print(ageTrain)
ageLable=ageTrain[:,2]#单挑出此处age那一列
# print(ageLable)
ageTrain=np.delete(ageTrain,2,axis=1)
ageTest=np.delete(ageTest,2,axis=1)

#预测Age这里选择GradientBoostingRegressor和MLPRegressor进行预测之后取平均，重复三次之后再取平均作为最终Age的预测结果。：
print('预测测试集Age:')
model1 = GradientBoostingRegressor(alpha=0.9, criterion='friedman_mse', init=None,
                                  learning_rate=0.03, loss='huber', max_depth=15,
                                  max_features='sqrt', max_leaf_nodes=None,
                                  min_impurity_decrease=0.0,
                                  min_samples_leaf=10, min_samples_split=40,
                                  min_weight_fraction_leaf=0.0, n_estimators=300,
                                  random_state=10, subsample=0.8, verbose=0,
                                  warm_start=False)#创建mlp神经网络对象
model2=MLPRegressor(activation='tanh', learning_rate='adaptive')
age_sum = []

for i in range(0,3):
    print(i,'th training:')
    model1.fit(ageTrain,ageLable)#模型训练
    age_model1 = model1.predict(ageTest)#模型预测
    model2.fit(ageTrain,ageLable)#模型训练
    age_model2 = model2.predict(ageTest)#模型预测
    age_sum.append(age_model1*0.5+age_model2*0.5)

age_model=[]
for i in range(len(ageTest)):
    asum=0
    for j in range(0,3):
        asum=asum+age_sum[j][i]
    age_model.append(asum/3)
print(age_model)

#把求出来的age填回去：
#先把空值的位置找出来：
nullIndex=[]
for i in range(0,len(isNull)):
    if isNull[i]==True:
        nullIndex.append(i)
for i in range(0,len(nullIndex)):
    testReview[nullIndex[i],2]=age_model[i]

#去除离群点：
rowLen=trainReview.shape[1]
shallDel=[]
for i in range(0,len(trainReview)):shallDel.append(0)
for j in range(0,rowLen):
    min=np.percentile(trainReview[:,j],6)
#     np.percentile(a, q, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
#
#     a: array，用来算分位数的对象，可以是多维的数组
#     q: 介于0 - 100
#     的float，用来计算是几分位的参数，如四分之一位就是25，如要算两个位置的数就(25, 75)
#     axis: 坐标轴的方向，一维的就不用考虑了，多维的就用这个调整计算的维度方向，取值范围0 / 1
#     out: 输出数据的存放对象，参数要与预期输出有相同的形状和缓冲区长度
#     overwrite_input: bool，默认False，为True时及计算直接在数组内存计算，计算后原数组无法保存
#     interpolation: 取值范围
#     {'linear', 'lower', 'higher', 'midpoint', 'nearest'}
#     默认liner，比如取中位数，但是中位数有两个数字6和7，选不同参数来调整输出
# keepdims: bool, 默认False, 为真时取中位数的那个轴将保留在结果中
    max = np.percentile(trainReview[:, j], 94)
    for i in range(0, len(trainReview)):
        if (trainReview[i,j]<min) or (trainReview[i,j]>max):
            shallDel[i]=1
for i in range(len(trainReview)-1,-1,-1):
    if shallDel[i]==1:
        trainReview=np.delete(trainReview,i,axis=0)
        trainLable = np.delete(trainLable, i, axis=0)


print('建模:')
model =LogisticRegression()
model.fit(trainReview, trainLable)
print('预测:')
pred_model = model.predict(testReview)
score = metrics.accuracy_score(testLable, pred_model)
matrix = metrics.confusion_matrix(testLable, pred_model)
print('>>>准确率\n', score)
print('\n>>>混淆矩阵\n', matrix)


