#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['figure.figsize'] = (5, 5)


# In[2]:


data=pd.read_csv('dataset\stroke.csv')


# In[3]:


data


# In[ ]:





# # Exploratory data analysis

# In[4]:


data.shape


# In[5]:


data.info()


# In[6]:


data.isnull().sum()


# # Lets fill Null Values

# In[7]:


data['bmi'].value_counts()


# In[8]:


data['bmi'].describe()


# In[9]:


data['bmi'].fillna(data['bmi'].mean(),inplace=True)


# In[10]:


data['bmi'].describe()


# In[11]:


data.isnull().sum()


# In[ ]:





# In[12]:


data.drop('id',axis=1,inplace=True)


# In[13]:


data


# In[ ]:





# # Outlier Removation

# In[14]:


from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=800, facecolor='w', edgecolor='k')


# In[15]:


data.plot(kind='box')
plt.show()


# # Label Encoding

# In[16]:


data.head()


# In[17]:


from sklearn.preprocessing import LabelEncoder
enc=LabelEncoder()


# In[18]:


gender=enc.fit_transform(data['gender'])


# In[19]:


smoking_status=enc.fit_transform(data['smoking_status'])


# In[20]:


work_type=enc.fit_transform(data['work_type'])
Residence_type=enc.fit_transform(data['Residence_type'])
ever_married=enc.fit_transform(data['ever_married'])


# In[21]:


data['work_type']=work_type


# In[22]:


data['ever_married']=ever_married
data['Residence_type']=Residence_type
data['smoking_status']=smoking_status
data['gender']=gender


# In[ ]:





# In[23]:


data


# In[24]:


data.info()


# In[ ]:





# In[ ]:





# # Splitting the data for train and test

# X ---train_X,test_X  80/20                   
# Y ---train_Y,test_Y

# In[25]:


X=data.drop('stroke',axis=1)


# In[26]:


X.head()


# In[27]:


Y=data['stroke']


# In[28]:


Y


# In[29]:


Y.max()


# In[30]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test=train_test_split(X,Y,test_size=0.2,random_state=0)


# In[31]:


X_train


# In[32]:


Y_train


# In[33]:


Y_train.max()


# In[34]:


X_test


# In[35]:


Y_test


# In[36]:


Y_test.max()


# # Normalize

# In[37]:


data.describe()


# In[ ]:





# In[38]:


from sklearn.preprocessing import StandardScaler
std=StandardScaler()


# In[39]:


X_train_std=std.fit_transform(X_train)
X_test_std=std.transform(X_test)


# # lets save the scaler object

# In[40]:


import pickle
import os


# In[41]:


scaler_path=os.path.join('models/scaler.pkl')
with open(scaler_path,'wb') as scaler_file:
    pickle.dump(std,scaler_file)


# In[ ]:





# In[42]:


X_train_std


# In[43]:


X_test_std


# In[ ]:





# # Training

# In[ ]:





# # Decision Tree

# In[44]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()


# In[45]:


dt.fit(X_train_std,Y_train)


# In[46]:


dt.feature_importances_


# In[47]:


X_train.columns


# In[48]:


Y_pred_dt=dt.predict(X_test_std) ## X_test_std


# In[ ]:





# In[49]:


Y_pred_dt.max()


# In[50]:


from sklearn.metrics import accuracy_score


# In[51]:


ac_dt=accuracy_score(Y_test,Y_pred_dt)


# In[52]:


ac_dt


# In[53]:


import joblib
model_path=os.path.join('models/dt.sav')
joblib.dump(dt,model_path)


# In[ ]:





# # Logistic Regression

# In[54]:


from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()


# In[55]:


lr.fit(X_train_std,Y_train)


# In[56]:


Y_pred_lr=lr.predict(X_test_std)


# In[57]:


Y_pred_lr.max()


# In[58]:


ac_lr=accuracy_score(Y_test,Y_pred_lr)


# In[59]:


ac_lr


# # KNN

# In[60]:


from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier()


# In[61]:


knn.fit(X_train_std,Y_train)


# In[62]:


Y_pred_knn=knn.predict(X_test_std)


# In[63]:


Y_pred_knn.max()


# In[64]:


ac_knn=accuracy_score(Y_test,Y_pred_knn)


# In[65]:


ac_knn


# In[66]:


# import joblib
# model_path=os.path.join('C:/Users/my pc/Desktop/ML/Stroke-Risk-Prediction-using-Machine-Learning-master/','models/knn.sav')
# joblib.dump(knn,model_path)


# # Random Forest

# In[67]:


from sklearn.ensemble import RandomForestClassifier
rf=RandomForestClassifier()


# In[68]:


rf.fit(X_train_std,Y_train)


# In[69]:


Y_pred_rf=rf.predict(X_test_std)


# In[70]:


Y_pred_rf.max()


# In[71]:


ac_rf=accuracy_score(Y_test,Y_pred_rf)


# In[72]:


ac_rf


# In[73]:


ac_knn


# In[74]:


ac_dt


# In[75]:


ac_lr


# In[ ]:





# In[ ]:





# # SVM

# In[76]:


from sklearn.svm import SVC


# In[77]:


sv=SVC()


# In[78]:


sv.fit(X_train_std,Y_train)


# In[79]:


Y_pred_sv=sv.predict(X_test_std)


# In[80]:


Y_pred_sv.max()


# In[81]:


ac_sv=accuracy_score(Y_test,Y_pred_sv)


# In[82]:


ac_sv


# In[83]:


ac_lr


# # Gradient Boosting

# In[84]:


from sklearn.ensemble import GradientBoostingClassifier


# In[85]:


gb = GradientBoostingClassifier()


# In[86]:


gb.fit(X_train_std,Y_train)


# In[87]:


Y_pred_gb=gb.predict(X_test_std)


# In[88]:


Y_pred_gb.max()


# In[89]:


ac_gb=accuracy_score(Y_test,Y_pred_gb)
ac_gb


# #  Stochastic Gradient Descent

# In[90]:


from sklearn.linear_model import SGDClassifier


# In[91]:


sgd = SGDClassifier()


# In[92]:


sgd.fit(X_train_std,Y_train)


# In[93]:


Y_pred_sgd=sgd.predict(X_test_std)


# In[94]:


Y_pred_sgd.max()


# In[95]:


ac_sgd=accuracy_score(Y_test,Y_pred_sgd)
ac_sgd


# # Naive Bayes

# In[ ]:





# # Gaussian NB

# In[96]:


from sklearn.naive_bayes import GaussianNB


# In[97]:


gnb = GaussianNB()


# In[98]:


gnb.fit(X_train_std,Y_train)


# In[99]:


Y_pred_gnb=gnb.predict(X_test_std)


# In[100]:


Y_pred_gnb.max()


# In[101]:


ac_gnb=accuracy_score(Y_test,Y_pred_gnb)
ac_gnb


# # Multinomial NB

# In[102]:


from sklearn.naive_bayes import MultinomialNB


# In[103]:


mnb = MultinomialNB()
mnb.fit(X_train,Y_train)


# In[104]:


Y_pred_mnb=mnb.predict(X_test_std)
Y_pred_mnb.max()


# In[105]:


ac_mnb=accuracy_score(Y_test,Y_pred_mnb)
ac_mnb


# # Bernoulli NB

# In[106]:


from sklearn.naive_bayes import BernoulliNB


# In[107]:


bnb = BernoulliNB()
bnb.fit(X_train_std,Y_train)


# In[108]:


Y_pred_bnb=bnb.predict(X_test_std)
Y_pred_bnb.max()


# In[109]:


ac_bnb=accuracy_score(Y_test,Y_pred_bnb)
ac_bnb


# # Complement NB

# In[110]:


from sklearn.naive_bayes import ComplementNB


# In[111]:


cnb = ComplementNB()
cnb.fit(X_train,Y_train)


# In[112]:


Y_pred_cnb=cnb.predict(X_test_std)
Y_pred_cnb.max()


# In[113]:


ac_cnb=accuracy_score(Y_test,Y_pred_cnb)
ac_cnb


# # XG Boost

# In[114]:


from xgboost import XGBClassifier


# In[115]:


xgb = XGBClassifier()
xgb.fit(X_train_std,Y_train)


# In[116]:


Y_pred_xgb=xgb.predict(X_test_std)
Y_pred_xgb.max()


# In[117]:


ac_xgb=accuracy_score(Y_test,Y_pred_xgb)
ac_xgb


# In[ ]:





# In[ ]:





# In[ ]:





# In[118]:


from matplotlib.pyplot import figure
figure(figsize=(14, 8), dpi=80)
algorithms = ['Decision Tree','Logistic','KNN','Random Forest','SVM','Gradient Boosting','SGD','GNB','MNB','BNB','CNB','XGB']
accuracy = [ac_dt,ac_lr,ac_knn,ac_rf,ac_sv,ac_gb,ac_sgd,ac_gnb,ac_mnb,ac_bnb,ac_cnb,ac_xgb]
plt.bar(algorithms,accuracy)
plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
# for i, v in enumerate(accuracy):
#     plt.text(i-.25, 
#               v/accuracy[i]+100, 
#               accuracy[i], 
#               fontsize=18, 
             
plt.show()


# In[ ]:





# In[119]:


Y_pred_dt.max()


# In[120]:


print(Y_pred_dt.max(),
Y_pred_sv.max(),
Y_pred_rf.max(),
Y_pred_lr.max(),
Y_pred_knn.max(),
 Y_pred_gb.max(),
 Y_pred_sgd.max(),
 Y_pred_gnb.max(),
Y_pred_mnb.max(),
 Y_pred_bnb.max(),
 Y_pred_cnb.max(),
 Y_pred_xgb.max(),)


# In[121]:


Y_pred_knn.max()


# In[ ]:





# In[ ]:





# In[122]:


X


# In[123]:


data


# In[124]:


import numpy as np
testtt = np.array([[0,44.0,0,0,1,0,1,85.28,26.200000,0]])


# In[125]:




# In[ ]:





# In[ ]:





# In[ ]:




