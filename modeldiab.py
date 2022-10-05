import numpy as np
import pandas as pd

import pickle
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv('kaggle_diabetes.csv')


df.isnull().sum()

df = df.drop_duplicates()

df['Glucose'] = df['Glucose'].replace(0,df['Glucose'].mean())
df['BloodPressure'] = df['BloodPressure'].replace(0,df['BloodPressure'].mean())
df['SkinThickness'] = df['SkinThickness'].replace(0,df['SkinThickness'].mean())
df['Insulin'] = df['Insulin'].replace(0,df['Insulin'].mean())
df['BMI'] = df['BMI'].replace(0,df['BMI'].mean())

cdf = df[['Pregnancies'	,'Glucose'	,'BloodPressure'	,'SkinThickness','Insulin'	,'BMI'	,'DiabetesPedigreeFunction'	,'Age'	,'Outcome']]

x = cdf.iloc[:,:-1].values
y = cdf.iloc[:,-1].values

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion = 'entropy')
rf.fit(x,y)

pickle.dump(rf, open('modeldiab.pkl','wb'))
modeldiab = pickle.load(open('modeldiab.pkl','rb'))
print(modeldiab.predict([[0,137,40,35,168,43.1,2.288,33]]))
