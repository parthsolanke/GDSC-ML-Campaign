import numpy as np
import pickle
from flask import Flask, request, render_template

app = Flask(__name__, static_folder='static')

# loading model and scaler
model = pickle.load(open(r'Model&Scaler\boston.pickle', 'rb'))
scaler = pickle.load(open(r'Model&Scaler\scalar.pickle', 'rb'))

def preprocess(data):
    '''
    Preprocessing the data
    '''
    data = np.array(data)
    return scaler.transform(data.reshape(1, -1))

# home page
@app.route('/')
def home():
    return render_template('index.html')

# prediction api
@app.route('/predict', methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # getting the data from the form
    input = [float(x) for x in request.form.values()]
    # preprocessing the data
    data = preprocess(input)
    # making prediction
    prediction = model.predict(data)
    output = round(prediction[0], 2)
    return render_template('predict.html', prediction_text=f'Predicted House Price ₹{output*80}')

if __name__ == "__main__":
    app.run(debug=True)