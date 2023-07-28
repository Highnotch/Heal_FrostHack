#Import Libraries
import glob
from keras.models import Sequential, load_model
import numpy as np
import pandas as pd
import keras as k
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
import matplotlib.pyplot as plt
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

df = pd.read_csv("dataset/kidney_disease.csv")

for column in df.columns:
        if df[column].dtype == np.number:
            continue
        df[column] = LabelEncoder().fit_transform(df[column])

df=df.fillna(0)
#Split the data
X = df[["sg", "al","sc","hemo","pcv","htn"]]
y = df["classification"]

#Split the data into 80% training and 20% testing 
X_train,  X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, shuffle=True)
print(X_train)
minmax = MinMaxScaler()
X_train = minmax.fit_transform(X_train)
X_test = minmax.transform(X_test)


import pickle
import os
scaler_path=os.path.join('models/scaler_kidney.pkl')
with open(scaler_path,'wb') as scaler_file:
    pickle.dump(minmax,scaler_file)

    
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(criterion='gini',n_estimators=100)
rf.fit(X_train,y_train)

from sklearn import model_selection
kfold = model_selection.KFold(n_splits=10)
scoring = 'accuracy'

acc_rf = cross_val_score(estimator = rf, X = X_train, y = y_train, cv = kfold,scoring=scoring)
print(acc_rf.mean())

import joblib
model_path=os.path.join('models/rf_kidney.sav')
joblib.dump(rf,model_path)