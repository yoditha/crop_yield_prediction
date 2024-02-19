import flask
from flask import Flask, render_template, request, send_static_file
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load('crpyield.pkl')

@app.route("/")
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        f1 = float(request.form['f1'])
        f2 = float(request.form['f2'])
        f3 = float(request.form['f3'])
        f4 = float(request.form['f4'])
        f5 = float(request.form['f5'])
        f6 = float(request.form['f6'])
        f7 = float(request.form['f7'])
        feature_array = [f1, f2, f3, f4, f5, f6, f7]
        feature = np.array(feature_array).reshape(1, -1)
        prediction = model.predict(feature)
        dic = {'rice':0, 'maize':1, 'chickpea':2, 'kidneybeans':3, 'pigeonpeas':4,
       'mothbeans':5, 'mungbean':6, 'blackgram':7, 'lentil':8, 'pomegranate':9,
       'banana':10, 'mango':11, 'grapes':12, 'watermelon':13, 'muskmelon':14, 'apple':15,
       'orange':16, 'papaya':17, 'coconut':18, 'cotton':19, 'jute':20, 'coffee':21}
        for key, value in dic.items():
            if value == prediction[0]:
                x = key

        return render_template('index.html', prediction='Predicted Crop: {}'.format(x))
@app.route('/image')
def get_image():
    filename='background.jpg'
    return app.send_static_file(filename)

if __name__ == "__main__":
    app.run() 
