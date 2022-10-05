import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('liver_disease_584.csv')

df["Albumin_and_Globulin_Ratio"] = df.Albumin_and_Globulin_Ratio.fillna(df['Albumin_and_Globulin_Ratio'].mean())
df[['Gender']] = df[['Gender']].replace(to_replace={'Male':1,'Female':0})

cdf = df[['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','Dataset']]

x = cdf.iloc[:,:-1].values
y = cdf.iloc[:,-1].values

from sklearn.ensemble import ExtraTreesClassifier
et = ExtraTreesClassifier(random_state=123)
et.fit(x, y)

pickle.dump(et, open('modelliv.pkl','wb'))
modelliv = pickle.load(open('modelliv.pkl','rb'))
print(modelliv.predict([[65,	0,	0.7,	0.1,	187,	16,	18,	6.8,	3.3,	0.90]]))
