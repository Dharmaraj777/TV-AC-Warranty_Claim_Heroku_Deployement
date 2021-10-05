
import numpy as np
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import pickle
import joblib
from flask import Flask, request, jsonify, render_template
import os
 #app name
app = Flask(__name__)
#load the saved model
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home(): 
    return render_template('home.html') 

@app.route('/predict',methods=['POST'])
def predict():  

    labels = ['Region', 'State','Area','City', 'Consumer_profile','Product_category',
              'Product_type','AC_1001_Issue','AC_1002_Issue','AC_1003_Issue','TV_2001_Issue','TV_2002_Issue',
              'TV_2003_Issue','Claim_Value','Service_Centre','Product_Age','Purchased_from','Call_details','Purpose'] 
    
    int_features = [float(x)  for x in request.form.values()]
    final_features = [np.array(int_features)]
    
    values = [np.array(final_features)] 
    prediction = model.predict(values) 
    result = labels[prediction[0]] 
    return render_template('home.html', output = 'The Claim is{}'.format(result))

if __name__ == "__main__":     
    app.run(debug=True)