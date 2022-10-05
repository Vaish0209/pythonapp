import numpy as np
import pandas as pd
import pickle

df = pd.read_csv('new_breast_cancer.csv')

df.drop('id', axis = 1, inplace = True)

cdf = df[['radius_mean',	'texture_mean',	'perimeter_mean',	'area_mean',	'smoothness_mean',	'compactness_mean',	'concavity_mean',	'concave points_mean',	'symmetry_mean',	'fractal_dimension_mean',	'radius_se',	'texture_se',	'perimeter_se',	'area_se',	'smoothness_se',	'compactness_se',	'concavity_se',	'concave points_se',	'symmetry_se',	'fractal_dimension_se',	'radius_worst',	'texture_worst',	'perimeter_worst',	'area_worst',	'smoothness_worst',	'compactness_worst',	'concavity_worst',	'concave points_worst',	'symmetry_worst',	'fractal_dimension_worst', 'diagnosis']]

x = cdf.iloc[:,:-1].values
y = cdf.iloc[:,-1].values

from sklearn.svm import SVC
sv = SVC()
sv.fit(x,y)


pickle.dump(sv, open('modelbreast.pkl','wb'))
modelbreast = pickle.load(open('modelbreast.pkl','rb'))
print(modelbreast.predict([[17.99,	10.38,	122.80,	1001.0,	0.11840,	0.27760,	0.3001,	0.14710,	0.2419,	0.07871,	1.0950,	0.9053,	8.589,	153.40,	0.006399,	0.04904,	0.05373,	0.01587,	0.03003,	0.006193,	25.38,	17.33,	184.60,	2019.0,	0.1622,	0.6656,	0.7119,	0.2654,	0.4601,	0.11890]]))
