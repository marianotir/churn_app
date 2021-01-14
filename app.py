# -*- coding: utf-8 -*-
"""
Created on Wed Jan  6 13:40:40 2021

@author: Mariano
"""

#-------------------------------
# Import libraries
#-------------------------------

# Deploy model libraries
import streamlit as st
import joblib

# Dataframe manipulation libraries
import pandas as pd


from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier

#-------------------------------
# StreamLit Application
#-------------------------------

# Load model 
model = joblib.load('model_14012021.pkl')

# Load Scaler
scaler = joblib.load('scaler_14012021.pkl')


def main():

    # Aplication header
    st.write("""
             # Chrun Prediction App 
             """)
             
    # Sidebar parameters
    st.sidebar.header('User input parameters')
    
    # Parameters
    Age = st.sidebar.slider('Age', 15, 30, 95)
    Balance = st.sidebar.slider('Balance', 0, 75000, 250000)
    NumOfProducts = st.sidebar.slider('NumOfProducts', 1, 2, 4)
    data = {'Age': Age,
            'Balance': Balance,
            'NumOfProducts': NumOfProducts}
    
    # Data for prediction output
    df = pd.DataFrame(data,index=[0])
    
    st.subheader(' Data input parameters')
    st.write(df)
    
    
    # Generate predictions on unseen data
    X_outsample = df.values
    
    X_outsample = scaler.transform(X_outsample)
    
    predictions = model.predict(X_outsample)
    
    prediction_output = predictions[0]

    if prediction_output > 0:
        prediction_output = 'Leaving probable'
    else: 
        prediction_output = 'Staying probable'
    st.subheader(' Prediction Output')
    st.write(prediction_output)
    
    
if __name__=='__main__':
    main()












