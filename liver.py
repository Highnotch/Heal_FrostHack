import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
data = pd.read_csv('dataset/indian_liver_patient.csv')

data['Albumin_and_Globulin_Ratio'] = data['Albumin_and_Globulin_Ratio'].fillna(data['Albumin_and_Globulin_Ratio'].mean())

def binary_encode(df, column, positive_value):
    df = df.copy()
    df[column] = df[column].apply(lambda x: 1 if x == positive_value else 0)
    return df

data = binary_encode(data, 'Gender', 'Male')

data = binary_encode(data, 'Dataset', 1)

y = data['Dataset']
X = data.drop('Dataset', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,random_state=0)

std=StandardScaler()
X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)

import pickle
import os
scaler_path=os.path.join('models/scaler_liver.pkl')
with open(scaler_path,'wb') as scaler_file:
    pickle.dump(std,scaler_file)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn=KNeighborsClassifier()
knn.fit(X_train_std,y_train)

Y_pred_knn=knn.predict(X_test_std)

ac_knn=accuracy_score(y_test,Y_pred_knn)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(X_train_std,y_train)

Y_pred_lr=lr.predict(X_test_std)
ac_lr=accuracy_score(y_test,Y_pred_lr)


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(X_train_std,y_train)

Y_pred_dt=dt.predict(X_test_std)
ac_dt=accuracy_score(y_test,Y_pred_dt)


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()
rf.fit(X_train_std,y_train)

Y_pred_rf=rf.predict(X_test_std)
ac_rf=accuracy_score(y_test,Y_pred_rf)


from sklearn.svm import SVC
sv=SVC()
sv.fit(X_train_std,y_train) 

Y_pred_sv=sv.predict(X_test_std)
ac_sv=accuracy_score(y_test,Y_pred_sv)
ac_sv

import joblib
model_path=os.path.join('models/sv.sav')
joblib.dump(sv,model_path)

from xgboost import XGBClassifier
xgb = XGBClassifier()
xgb.fit(X_train_std,y_train)

Y_pred_xgb=xgb.predict(X_test_std)
ac_xgb=accuracy_score(y_test,Y_pred_xgb)


from sklearn.linear_model import SGDClassifier
sgd = SGDClassifier()
sgd.fit(X_train_std,y_train)

Y_pred_sgd=sgd.predict(X_test_std)
ac_sgd=accuracy_score(y_test,Y_pred_sgd)



from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X_train_std,y_train)

Y_pred_gnb=gnb.predict(X_test_std)
ac_gnb=accuracy_score(y_test,Y_pred_gnb)


from sklearn.naive_bayes import BernoulliNB
bnb = BernoulliNB()
bnb.fit(X_train_std,y_train)

Y_pred_bnb=bnb.predict(X_test_std)
ac_bnb=accuracy_score(y_test,Y_pred_bnb)
