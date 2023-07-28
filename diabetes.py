import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import  accuracy_score, f1_score, precision_score,confusion_matrix, recall_score, roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from IPython.display import Image

df = pd.read_csv('dataset/diabetes_data_upload.xls')
df['class'] = df['class'].apply(lambda x: 0 if x=='Negative' else 1)
X= df.drop(['class'],axis=1)
y=df['class']
X=X.drop(['weakness','Polyphagia','Genital thrush','visual blurring','muscle stiffness','Obesity'],axis=1)
objList = X.select_dtypes(include = "object").columns
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()

for feat in objList:
    X[feat] = le.fit_transform(X[feat].astype(str))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,stratify=y, random_state = 0)
minmax = MinMaxScaler()
X_train = minmax.fit_transform(X_train)
X_test = minmax.transform(X_test)
import pickle
import os
scaler_path=os.path.join('models/scaler_diabetes.pkl')
with open(scaler_path,'wb') as scaler_file:
    pickle.dump(minmax,scaler_file)
rf = RandomForestClassifier(criterion='gini',n_estimators=100)
rf.fit(X_train,y_train)
import joblib
model_path=os.path.join('models/rf_diabetes.sav')
joblib.dump(rf,model_path)

from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'

acc_rf = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = kfold,scoring=scoring)
acc_rf.mean()

y_predict = rf.predict(X_test)
c = 0
for i in range(104):
    if(y_predict[i]==y_test[y_test.index[i]]):
        c+=1
