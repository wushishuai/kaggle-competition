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

df = pd.DataFrame({ 'a': [3,5,np.NaN,5,np.NaN,4,3,7],
                    'b': [6,7,8,9,0,2,3,np.NaN],
                    'c': [3,2,1,5,6,5,4,3]})
print(df)
df.dropna( subset=['a'], inplace=True )

df['a']=df['a'].fillna(df['a'].dropna().mode()[0])
print(df)