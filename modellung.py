import numpy as np
import pandas as pd

import pickle
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("survey lung cancer.csv")

df = df.drop_duplicates()

cdf = df[['GENDER','AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE','CHRONIC DISEASE','FATIGUE ','ALLERGY ','WHEEZING','ALCOHOL CONSUMING','COUGHING','SHORTNESS OF BREATH','SWALLOWING DIFFICULTY','CHEST PAIN','LUNG_CANCER']]

X = cdf.iloc[:, :-1].values
y = cdf.iloc[:, -1].values

from sklearn.model_selection import train_test_split
# separating into train and testing
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify=y,random_state=1)

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split

train_X, val_X, train_y, val_y = train_test_split(X,y,random_state = 1)

lung_cancer_model = DecisionTreeRegressor(random_state=1)

# Fit Lung Cancer Model with the training data.
lung_cancer_model.fit(train_X, train_y)

val_predictions = lung_cancer_model.predict(val_X)

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor


from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
dtc = DecisionTreeClassifier()
dtc.fit(train_X,train_y)

y_pred = dtc.predict(val_X)

pickle.dump(dtc, open('modellung.pkl','wb'))
model = pickle.load(open('modellung.pkl','rb'))
print(model.predict([[1,69,1,2,2,1,1,2,1,2,2,2,2,2,2]]))
