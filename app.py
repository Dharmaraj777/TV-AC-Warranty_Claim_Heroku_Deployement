
import numpy as np
import pickle
import joblib
from flask import Flask, request, jsonify, render_template
 #app name
app = Flask(__name__)
#load the saved model
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home(): 
    return render_template('home.html') 


def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1,19)
    loaded_model = pickle.load(open("model.pkl","rb"))
    result = loaded_model.predict(to_predict)
    return result[0]
    
@app.route('/predict',methods=['POST'])
def result():
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list=list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result = ValuePredictor(to_predict_list)
        
        if int(result)==1:
            prediction='The Claim is Fraud'
        else:
            prediction='The Claim is Genuine'
    
    return render_template('home.html', prediction=prediction)

if __name__ == "__main__":     
    app.run(debug=True)
    