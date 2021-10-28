# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 18:09:25 2021

@author: 91779
"""

import pandas as pd
import numpy as np
from sklearn import preprocessing
#from sklearn.model_selection import cross_val_score
#from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, label_binarize
#from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pickle

# this is the main function in which we define our webpage  
def main():       
    # front end elements of the web page 
    html_temp = """ 
    <div style ="background-color:yellow;padding:5px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Claim Prediction ML App</h1> 
    </div> 
    """
    
     # display the front end aspect
    st.markdown(html_temp, unsafe_allow_html = True) 

#st.title('Model Deployment: Claims Prediction - Dharma')

st.sidebar.header('User Input Parameters')

Warranty_claim = pd.read_csv("Warranty Claim.csv")
#Eliminating Unwanted columns
Warranty_claim.drop(Warranty_claim.columns[[0]], axis = 1, inplace = True)


# Declaring features & target
#X = Warranty_claim.drop(['Fraud'], axis=1)
#Y = Warranty_claim['Fraud']
# creating one hot encoding of the categorical columns.
#X = pd.get_dummies(X, columns =['Region', 'State', 'Area', 'City', 
 #                               'Consumer_profile', 'Product_category', 'Product_type',
 #                               'Service_Centre','Purchased_from','Purpose'])

def user_input_features():
    Region = st.sidebar.selectbox('Region',Warranty_claim.Region.unique())
    State = st.sidebar.selectbox('State',Warranty_claim.State.unique())
    Area = st.sidebar.selectbox('Area',Warranty_claim.Area.unique())
    City = st.sidebar.selectbox('City',Warranty_claim.City.unique())
    Consumer_profile = st.sidebar.selectbox('Consumer_profile',("Business","Personal"))
    Product_category = st.sidebar.selectbox('Product_category',("Entertainment","Household"))
    Product_type = st.sidebar.selectbox('Product_type',("TV","AC"))
    AC_1001_Issue = st.sidebar.selectbox('AC_1001_Issue',Warranty_claim.AC_1001_Issue.unique())
    AC_1002_Issue = st.sidebar.selectbox('AC_1002_Issue',Warranty_claim.AC_1002_Issue.unique())
    AC_1003_Issue = st.sidebar.selectbox('AC_1003_Issue',Warranty_claim.AC_1003_Issue.unique())
    TV_2001_Issue = st.sidebar.selectbox('TV_2001_Issue',Warranty_claim.TV_2001_Issue.unique())
    TV_2002_Issue = st.sidebar.selectbox('TV_2002_Issue',Warranty_claim.TV_2002_Issue.unique())
    TV_2003_Issue = st.sidebar.selectbox('TV_2003_Issue',Warranty_claim.TV_2003_Issue.unique())
    Claim_Value  = st.sidebar.number_input("Insert the claim value")
    Service_Centre = st.sidebar.selectbox('Service_Centre',Warranty_claim.Service_Centre.unique())
    Product_Age = st.sidebar.number_input("Insert the product age")
    Purchased_from = st.sidebar.selectbox('Purchased_from',("Manufacturer","Dealer","Internet"))
    Call_details = st.sidebar.number_input("Insert the call details")
    Purpose = st.sidebar.selectbox('Purpose',("Complaint","Claim","Other"))
    
        # Pre-processing user input    
    if Consumer_profile == "Business":
        Consumer_profile = 0
    else:
        Consumer_profile = 1
 
    if Product_category == "Entertainment":
        Product_category = 0
    else:
        Product_category = 1
        
    if Product_type == "AC":
        Product_type = 0
    else:
        Product_type = 1
        
    if Purchased_from == "Dealer":
       Purchased_from = 0
    else:
        Purchased_from = 1
        
    if Purchased_from == "Internet":
       Purchased_from = 1
    else:
        Purchased_from = 2
        
    if Purpose == "Claim":
       Purpose = 0
    else:
        Purpose = 1
        
    if Purpose == "Complaint":
       Purpose = 1
    else:
        Purpose = 2
    
    data = {'Region':Region,
            'State':State,
            'Area':Area,
            'City':City,
            'Consumer_profile':Consumer_profile,
            'Product_category':Product_category,
            'Product_type':Product_type,
            'AC_1001_Issue':AC_1001_Issue,
            'AC_1002_Issue':AC_1002_Issue,
            'AC_1003_Issue':AC_1003_Issue,
            'TV_2001_Issue':TV_2001_Issue,
            'TV_2002_Issue':TV_2002_Issue,
            'TV_2003_Issue':TV_2003_Issue,
            'Claim_Value':Claim_Value,
            'Service_Centre':Service_Centre,
            'Product_Age':Product_Age,
            'Purchased_from':Purchased_from,
            'Call_details':Call_details,
            'Purpose':Purpose,}
    features = pd.DataFrame(data,index = [0])
    return features 
    
df = user_input_features()
df.drop(df.columns[[0,1,2,3,]], axis = 1, inplace = True)
#label_encoder = preprocessing.LabelEncoder()
#df['Region']= label_encoder.fit_transform(df['Region'])
#df['State']= label_encoder.fit_transform(df['State'])
#df['Area']= label_encoder.fit_transform(df['Area'])
#df['City']= label_encoder.fit_transform(df['City'])
#df['Consumer_profile']= label_encoder.fit_transform(df['Consumer_profile'])
#df['Product_category']= label_encoder.fit_transform(df['Product_category'])
#df['Product_type']= label_encoder.fit_transform(df['Product_type'])
#df['Purchased_from']= label_encoder.fit_transform(df['Purchased_from'])
#df['Purpose']= label_encoder.fit_transform(df['Purpose'])
## Using PCA instead of eleminating columns
#from sklearn.decomposition import PCA
loaded_model = pickle.load(open('model.pkl','rb'))
#pca = PCA(n_components = 24)
#df2 = pca.fit_transform(np.array(df1))
#Resampling - SMOTE(Synthetic Minority Over Sampling Technique)
#from imblearn.over_sampling import SMOTE
#method = SMOTE(random_state = 7)
#X_resampled, Y_resampled = method.fit_resample(X_new,Y)
st.subheader('User Input parameters')
st.write(df)



## Using PCA instead of eleminating columns
#from sklearn.decomposition import PCA
#pca = PCA(n_components = 25)
#X_new = pca.fit_transform(X)
#Resampling - SMOTE(Synthetic Minority Over Sampling Technique)
#from imblearn.over_sampling import SMOTE
#method = SMOTE(random_state = 7)
#X_resampled, Y_resampled = method.fit_resample(X_new,Y)
# Random Forest Classification
#from sklearn.ensemble import RandomForestClassifier

#num_trees = 100
#max_features = 3
#kfold = KFold(n_splits=10, shuffle=False)
#Random_Forest = RandomForestClassifier(n_estimators=num_trees, max_features=max_features, criterion="entropy",random_state=7)
#results = cross_val_score(Random_Forest, X_resampled, Y_resampled, cv=kfold)
#print(results.mean())
#Selected Random Forest
#model = Random_Forest.fit(X_new,Y)
#RandomForestClassifier=Random_Forest.predict(X_new)
#from sklearn import metrics
#print(metrics.classification_report(Y, RandomForestClassifier))
#Accuracy Score
#from sklearn.metrics import accuracy_score

#accuracy_score(Y,RandomForestClassifier)

#pickle.dump(model, open('model.pkl','wb'))
# Loading model to compare the results
#model = pickle.load(open('model.pkl','rb'))

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.subheader('Prediction Probability')
st.write(prediction_proba)
if st.button("Predict"): 
   st.success('The Claim is Fraud' if prediction_proba[0][1] > 0.5 else 'The Claim is Genuine')

     
if __name__=='__main__': 
    main()
