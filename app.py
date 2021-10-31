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


def user_input_features():
    region_display = ('North','East','West','South','North East','North West','South East','South West')
    region_options = list(range(len(region_display)))
    Region = st.sidebar.selectbox("Region",region_options, format_func=lambda x:region_display[x])
    state_display = ('Karnataka', 'Haryana', 'Tamil Nadu', 'Jharkhand', 'Kerala',
                         'Andhra Pradesh', 'Bihar', 'Gujarat', 'Delhi', 'Maharashtra',
                         'West Bengal', 'Goa', 'Jammu and Kashmir', 'Assam', 'Rajasthan',
                         'Madhya Pradesh', 'Uttar Pradesh', 'Tripura', 'Himachal Pradesh',
                         'Orissa')
    state_options = list(range(len(state_display)))
    State = st.sidebar.selectbox("State",state_options, format_func=lambda x: state_display[x])
    area_display = ('Urban', 'Rural')
    area_options = list(range(len(area_display)))
    Area = st.sidebar.selectbox("Area",area_options, format_func=lambda x: area_display[x])
    city_display = ('Bangalore', 'Chandigarh', 'Chennai', 'Ranchi', 'Kochi',
                        'Hyderabad', 'Patna', 'Purnea', 'Vadodara', 'New Delhi', 'Mumbai',
                        'Ahmedabad', 'Pune', 'Kolkata', 'Vizag', 'Panaji', 'Srinagar',
                        'Guwhati', 'Jaipur', 'Bhopal', 'Meerut', 'Delhi', 'Agartala',
                        'Shimla', 'Bhubaneswar', 'Vijayawada', 'Lucknow')
    city_options = list(range(len(city_display)))
    City = st.sidebar.selectbox("City",city_options, format_func=lambda x: city_display[x])
    Consumer_profile = st.sidebar.selectbox('Consumer_profile',("Business","Personal"))
    Product_category = st.sidebar.selectbox('Product_category',("Entertainment","Household"))
    Product_type = st.sidebar.selectbox('Product_type',("TV","AC"))
    ## AC_1001_Issue
    AC_1001_Issue_display = ('No Issue','Repair','Replacement')
    AC_1001_Issue_options = list(range(len(AC_1001_Issue_display)))
    AC_1001_Issue = st.selectbox("AC 1001 Issue",AC_1001_Issue_options, format_func=lambda x: AC_1001_Issue_display[x])
        
    if AC_1001_Issue=='No Issue':
            AC_1001_Issue=0
    elif AC_1001_Issue=='Repair':
            AC_1001_Issue=1
    elif AC_1001_Issue=='Replacement':
            AC_1001_Issue=2

   ## AC_1002_Issue
    AC_1002_Issue_display = ('No Issue','Repair','Replacement')
    AC_1002_Issue_options = list(range(len(AC_1002_Issue_display)))
    AC_1002_Issue = st.selectbox("AC 1002 Issue",AC_1002_Issue_options, format_func=lambda x: AC_1002_Issue_display[x])
     
    if AC_1002_Issue=='No Issue':
            AC_1002_Issue=0
    elif AC_1002_Issue=='Repair':
            AC_1002_Issue=1
    elif AC_1002_Issue=='Replacement':
            AC_1002_Issue=2
   ## AC_1003_Issue
    AC_1003_Issue_display = ('No Issue','Repair','Replacement')
    AC_1003_Issue_options = list(range(len(AC_1003_Issue_display)))
    AC_1003_Issue = st.selectbox("AC 1002 Issue",AC_1003_Issue_options, format_func=lambda x: AC_1003_Issue_display[x])
     
    if AC_1003_Issue=='No Issue':
            AC_1003_Issue=0
    elif AC_1003_Issue=='Repair':
            AC_1003_Issue=1
    elif AC_1003_Issue=='Replacement':
            AC_1003_Issue=2
 ## TV_2001_Issue
    TV_2001_Issue_display = ('No Issue','Repair','Replacement')
    TV_2001_Issue_options = list(range(len(TV_2001_Issue_display)))
    TV_2001_Issue = st.selectbox("TV 2001 Issue",TV_2001_Issue_options, format_func=lambda x: TV_2001_Issue_display[x])
        
    if TV_2001_Issue=='No Issue':
            TV_2001_Issue=0
    elif TV_2001_Issue=='Repair':
            TV_2001_Issue=1
    elif TV_2001_Issue=='Replacement':
            TV_2001_Issue=2
             
 ## TV_2002_Issue
    TV_2002_Issue_display = ('No Issue','Repair','Replacement')
    TV_2002_Issue_options = list(range(len(TV_2002_Issue_display)))
    TV_2002_Issue = st.selectbox("TV 2002 Issue",TV_2002_Issue_options, format_func=lambda x: TV_2002_Issue_display[x])
    
    if TV_2002_Issue=='No Issue':
            TV_2002_Issue=0
    elif TV_2002_Issue=='Repair':
            TV_2002_Issue=1
    elif TV_2002_Issue=='Replacement':
            TV_2002_Issue=2
 ## TV_2003_Issue
    TV_2003_Issue_display = ('No Issue','Repair','Replacement')
    TV_2003_Issue_options = list(range(len(TV_2003_Issue_display)))
    TV_2003_Issue = st.selectbox("TV 2003 Issue",TV_2003_Issue_options, format_func=lambda x: TV_2003_Issue_display[x])
    if TV_2003_Issue=='No Issue':
            TV_2003_Issue=0
    elif TV_2003_Issue=='Repair':
            TV_2003_Issue=1
    elif TV_2003_Issue=='Replacement':
            TV_2003_Issue=2    
            
    Claim_Value  = st.sidebar.number_input("Insert the claim value")
    service_centre_display = [10, 12, 14, 16, 15, 13, 11]
    service_centre_options = list(range(len(service_centre_display)))
    Service_Centre = st.sidebar.selectbox("Service Centre Code",service_centre_options, format_func=lambda x: service_centre_display[x])
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
loaded_model = pickle.load(open('model.pkl','rb'))
st.subheader('User Input parameters')
st.write(df)

prediction = loaded_model.predict(df)
prediction_proba = loaded_model.predict_proba(df)

st.subheader('Predicted Result')
st.subheader('Prediction Probability')
st.write(prediction_proba)
if st.button("Predict"): 
   st.success('The Claim is Fraud' if prediction_proba[0][1] > 0.5 else 'The Claim is Genuine')

     
if __name__=='__main__': 
    main()