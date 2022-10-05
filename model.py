import numpy as np
import pandas as pd

import pickle
from sklearn.preprocessing import LabelEncoder


df = pd.read_csv('heart (1).csv')

df.isnull().sum()


df.head()


cdf = df[['age','sex','cp','trestbps','chol','fbs','restecg','thalach','exang','oldpeak','slope','ca','thal','target']]


x = cdf.iloc[:, :-1].values
y = cdf.iloc[:, -1].values

from sklearn.linear_model import LogisticRegression
log_classifier = LogisticRegression()

log_classifier.fit(x, y)

pickle.dump(log_classifier, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[53,1,0,140,203,1,0,155,1,3.1,0,0,3]]))
