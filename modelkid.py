# Importing libraries
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('kidney_disease.csv')

# dropping id column
df.drop('id', axis = 1, inplace = True)

# Mapping
df[['htn','dm','cad','pe','ane']] = df[['htn','dm','cad','pe','ane']].replace(to_replace={'yes':1,'no':0})
df[['rbc','pc']] = df[['rbc','pc']].replace(to_replace={'abnormal':1,'normal':0})
df[['pcc','ba']] = df[['pcc','ba']].replace(to_replace={'present':1,'notpresent':0})
df[['appet']] = df[['appet']].replace(to_replace={'good':1,'poor':0,'no':np.nan})
df['classification'] = df['classification'].replace(to_replace={'ckd':1.0,'ckd\t':1.0,'notckd':0.0,'no':0.0})

df['wc']=df['wc'].replace(["\t6200","\t8400","\t?"],[6200,8400, np.nan])
df['pcv']=df['pcv'].replace(["\t43","\t?"],[43,np.nan])
df['rc']=df['rc'].replace(["\t?"],[np.nan])

df = df.fillna(method='ffill')
df = df.fillna(method='backfill')

#changing object type to numeric
df['pcv']=df['pcv'].astype(int)
df['wc']=df['wc'].astype(int)
df['rc']=df['rc'].astype(float)

# Further cleaning
df['pe'] = df['pe'].replace(to_replace='good',value=0)
df['appet'] = df['appet'].replace(to_replace='no',value=0)
df['cad'] = df['cad'].replace(to_replace='\tno',value=0)
df['dm'] = df['dm'].replace(to_replace={'\tno':0,'\tyes':1,' yes':1, '':np.nan})

cdf = df[['age'	,'bp'	,'sg'	,'al','su'	,'rbc'	,'pc'	,'pcc'	,'ba' ,'bgr'	,'bu'	,'sc','sod'	,'pot'	,'hemo'	,'pcv'	,'wc', 'rc',	'htn',	'dm'	,'cad'	,'appet'	,'pe'	,'ane','classification']]

x = cdf.iloc[:,:-1].values
y = cdf.iloc[:,-1].values

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(x, y)

pickle.dump(rf, open('modelkid.pkl','wb'))
modelkid = pickle.load(open('modelkid.pkl','rb'))

print(modelkid.predict([[48.0	,80.0,	1.020,	1.0,	0.0,	0.0,	0.0,	0.0,	0.0,	121.0,	36.0,	1.2,	111.0,	2.5,	15.4,	44,	7800,	5.2,	1.0,	1,	0,	1.0,	0.0,	0.0]]))
