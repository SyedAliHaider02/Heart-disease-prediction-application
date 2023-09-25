from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import pickle


model = pickle.load(open('heart1.pkl', 'rb'))

app = Flask(__name__)

@app.route('/')
def man():
    return render_template('front.html')


@app.route('/predict', methods=['GET', 'POST'])
def home():
    data1 = request.form['age']
    data2 = request.form['gender']
    data3 = request.form['cp']
    data4 = request.form['tresbps']
    data5 = request.form['chol']
    data6 = request.form['fbs']
    data7 = request.form['restecg']
    data8 = request.form['thalach']
    data9 = request.form['exang']
    data10 = request.form['oldpeak']
    data11 = request.form['slope']
    data12 = request.form['ca']
    data13 = request.form['thal']
    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12, data13]])
    pred = model.predict(arr)
    return render_template('after.html', data=pred)

    

if __name__=="__main__":
    app.run(debug=True)
