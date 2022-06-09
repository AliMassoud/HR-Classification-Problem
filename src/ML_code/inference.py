import numpy as np
import pandas as pd
import src.ML_code.preprocess as preprocess_py
import joblib


model_name = 'src/ML_code/Models/model.sav'


def make_predictions(input_data: pd.DataFrame) -> np.ndarray:
    if('Metier' in input_data.columns.values):
        X = input_data.sort_index(axis=1).drop(['Metier'], axis=1)
    else:
        X = input_data
    X = preprocess_py.before_split_data_type(X)

    Ordinal, numeric = preprocess_py.split_(X)
    x_test = preprocess_py.preprocess(Ordinal, numeric, dataset_Typee=True)
    loaded_model = joblib.load(model_name)
    y_pred = loaded_model.predict(x_test)
    return y_pred