# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 12:19:07 2021

@author: Mariano
"""

#------------------------
# Load Initial Libraries
#------------------------

import pandas as pd 
import numpy as np
from sklearn.compose import make_column_transformer


#----------------------
# Load data
#----------------------

df = pd.read_csv('C:/Users/Mariano/DS_Models/St_Chrun/Churn_Modelling.csv')  


#-------------------------
# Preprocess Data
#-------------------------

# Delete unnecesary variables
df = df.drop(columns=['RowNumber','CustomerId','Surname','Tenure',
                      'Geography','Gender','IsActiveMember'])

# Check apareance
df.head(10)

# description
df.describe()

# Check column types
df.dtypes

# Check shape
df.shape


#-----------------------------
# Feature selection
#-----------------------------

# Target and feats variables
from sklearn import preprocessing

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

le = preprocessing.LabelEncoder()
le_fit = le.fit(Y)
target_encoded_le = le_fit.transform(Y)


# Split train and test dataset 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, target_encoded_le, test_size=0.2)

# Scale data 
from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
X_train = mm_scaler.fit_transform(X_train)

# Simple model for fetaure selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import SelectFromModel
clf = DecisionTreeClassifier()
clf.fit(X_train, Y_train)

trans = SelectFromModel(clf)
X_trans = trans.fit_transform(X_train, Y_train)

print("We started with {0} features but retained only {1} of them!".
      format(X_train.shape[1], X_trans.shape[1]))

columns_retained_FromMode = df.iloc[:, 1:].columns[trans.get_support()].values

print("Columns selected are: {0}".
      format(columns_retained_FromMode))


#-----------------------------
# Prepare data for training
#-----------------------------

# Define dataframe with the new columns 
df = df[columns_retained_FromMode]

# Target and feats variables
from sklearn import preprocessing

X = df.iloc[:,0:-1].values
Y = df.iloc[:,-1].values

le = preprocessing.LabelEncoder()
le_fit = le.fit(Y)
target_encoded_le = le_fit.transform(Y)

# Split data set again after feature selection 
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, target_encoded_le, test_size=0.2)

# Scale data 
from sklearn import preprocessing
mm_scaler = preprocessing.MinMaxScaler()
mm_scaler.fit(X_train)
X_train = mm_scaler.transform(X_train)
X_test = mm_scaler.transform(X_test)


#----------------------------
# Train model
#----------------------------
from sklearn.ensemble import RandomForestClassifier


model_rf = RandomForestClassifier(n_estimators = 50, random_state = 0)
model_rf.fit(X_train,Y_train)

pred_train = model_rf.predict(X_train)
pred       = model_rf.predict(X_test)

  # Checking accuracy
from sklearn.metrics import accuracy_score
print('accuracy_score =',accuracy_score(Y_test, pred))

  # f1 score
from sklearn.metrics import f1_score
print('f1_score =',f1_score(Y_test, pred))

  # recall score
from sklearn.metrics import recall_score
print('recall_score =',recall_score(Y_test, pred))

# Note: Since random forest model is applied there is no need for normalization or scale the data 
#       before training

#-----------------------w
# Save model
#-----------------------
import pickle
# save model
pkl_model = "C:/Users/Mariano/DS_Models/St_Chrun/model_14012021.pkl"
pickle.dump(model_rf, open(pkl_model, 'wb'))
    
# save the scaler
pkl_scaler = "C:/Users/Mariano/DS_Models/St_Chrun/scaler_14012021.pkl"
pickle.dump(mm_scaler, open(pkl_scaler, 'wb'))


#--------------------------
# Load the model
#--------------------------
import joblib

# Load model
pkl_model = "C:/Users/Mariano/DS_Models/St_Chrun/model_14012021.pkl"
model = joblib.load(pkl_model)

# Load scaler
pkl_scaler = "C:/Users/Mariano/DS_Models/St_Chrun/scaler_14012021.pkl"
scaler = joblib.load(pkl_scaler)

# Generate predictions 
Age = 25
Balance  = 3000
NumOfProducts = 1

data = {'Age': Age,
        'Balance': Balance,
        'NumOfProducts': NumOfProducts}
    
# Data for prediction output
df = pd.DataFrame(data,index=[0])
X_outsample = df.values

X_outsample = scaler.transform(X_outsample)
    
pred = model.predict(X_outsample)
pred





