from flask import Flask, request, json
from src.ML_code.inference import make_predictions
import pandas as pd
import numpy as np
y_encoded = {'0': 'Data architecte','1': 'Data engineer', '2': 'Data scientist', '3': 'Lead data scientist'}

app = Flask(__name__)

@app.route('/SubmitFile', methods=["GET"])
def index():
    saved_file = request.files['data_file']
    df = pd.read_csv(saved_file, sep=';')
    result = make_predictions(df)
    df1 = pd.DataFrame({'Result':result})
    df1['Result'].replace({0:y_encoded['0'], 1:y_encoded['1'], 2:y_encoded['2'], 3:y_encoded['3']}, inplace=True)
    df['Metier'] = df1['Result']
    print(df)
    return json.dumps(df.to_dict())


@app.route("/Submit", methods=['GET'])
def submit():
    data = json.loads(request.data)
    temp = np.array([data['Entreprise'],data['Technologies'],data['Diplome'], data['Experience'], data['Ville']])
    my_data = pd.DataFrame([temp], columns=['Entreprise','Technologies', 'Diplome','Experience', 'Ville'])
    result = make_predictions(my_data) # Pass my data to the model
    
    df = pd.DataFrame({'Result':result})
    df['Result'].replace({0:y_encoded['0'], 1:y_encoded['1'], 2:y_encoded['2'], 3:y_encoded['3']}, inplace=True)
    return {
    "Entreprise": my_data['Entreprise'].values[0],
    "Technologies": my_data['Technologies'].values[0],
    "Diplome": my_data['Diplome'].values[0],
    "Experience": my_data['Experience'].values[0],
    "Ville": my_data['Ville'].values[0],
    "Prediction": df['Result'][0]
    }



