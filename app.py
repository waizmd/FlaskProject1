from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('model.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    age = int(request.form['age'])
    experience = int(request.form['experience'])
    prediction = model.predict([[age, experience]])[0]
    return render_template('index.html', prediction=round(prediction, 2))

if __name__ == "__main__":
    app.run(debug=True)