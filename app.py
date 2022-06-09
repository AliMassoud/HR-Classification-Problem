from flask import Flask, request, json
from src.ML_code.inference import make_predictions
import pandas as pd
import numpy as np
y_encoded = {'0': 'Data architecte','1': 'Data engineer', '2': 'Data scientist', '3': 'Lead data scientist'}

app = Flask(__name__)

@app.route("/Submit", methods=['GET'])
def submit():
    data = json.loads(request.data)
    temp = np.array([data['Entreprise'],data['Technologies'],data['Diplome'], data['Experience'], data['Ville']])
    my_data = pd.DataFrame([temp], columns=['Entreprise','Technologies', 'Diplome','Experience', 'Ville'])
    result = make_predictions(my_data) # Pass my data to the model
    
    df = pd.DataFrame({'Result':result})
    df['Result'].replace({0:y_encoded['0'], 1:y_encoded['1'], 2:y_encoded['2'], 3:y_encoded['3']}, inplace=True)

    # Percentage calculation    
    final = (df['Result'].value_counts()/df['Result'].count())*100
    # print(final)
    
    temp_df = final.to_frame()
    temp_df.sort_values(by='Result')
    print(temp_df)
    # print(temp_df['Result'][0])
    temp_df['Result'] = temp_df['Result'].astype(str) + '%'
    return {
    "Entreprise": my_data['Entreprise'].values[0],
    "Technologies": my_data['Technologies'].values[0],
    "Diplome": my_data['Diplome'].values[0],
    "Experience": my_data['Experience'].values[0],
    "Ville": my_data['Ville'].values[0],
    "Prediction": json.dumps(temp_df['Result'].to_dict())
    }



@app.route('/SubmitFile', methods=["GET"])
def index():
    if request.method == 'GET':
        saved_file = request.files['data_file']
        df = pd.read_csv(saved_file, sep=';')
        # g_list = dict()
        g = []
        result = make_predictions(df)
        print(result)
    return "Hello"