from sklearn.model_selection import train_test_split
import string
import numpy as np
import pandas as pd
import src.ML_code.preprocess as preprocess_py
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier



train_data_path_after_split = "Data/splitted_data/Train_after_split.csv"
test_data_path_after_split = "Data/splitted_data/Test_after_split.csv"
# val_data_path_after_split = "../../Data/splitted_data/val_after_split.csv"
model_name = 'src/ML_code/Models/model.sav'
def Save_(x: pd.DataFrame, y: pd.DataFrame, filepath: string):
    my_data = pd.concat([
        pd.DataFrame(x),
        pd.DataFrame(y)
    ], axis=1)
    my_data.to_csv(filepath)

def compute_metrics(y_pred, y_test):
    
    accuracy = round(accuracy_score(y_test, y_pred), 2)
    precision = round(precision_score(y_test, y_pred, average='micro'),2)
    recall = round(recall_score(y_test, y_pred, average='micro'),2)
    
    print('Accuracy (proportion of correct predictions) is ' + str(accuracy))
    print('Precision (proportion of true purchases among purchase predictions) is ' + str(precision))
    print('Recall (proportion of true purchases that are correctly predicted) is ' + str(recall))
    return pd.DataFrame({
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
    }, index=['RandomForest'])



def Tune_Random_Forest(X_train, y_train):
    estimator = RandomForestClassifier()
    param_grid = { 
            "n_estimators"      : [10,20,30],
            "max_features"      : ["sqrt", "log2"],
            "min_samples_split" : [2,4,8,16],
            "bootstrap": [True, False],
            }
    grid = GridSearchCV(estimator, param_grid, n_jobs=-1, cv=5)
    grid.fit(X_train, y_train)
    return grid.best_score_ , grid.best_params_



def build_model(data: pd.DataFrame) -> dict[str, str]:
    # clean data
    data = preprocess_py.before_split_data_type(data)
    Y = data['Metier']
    Y = preprocess_py.encode_target(Y)
    X = data.sort_index(axis=1).drop(['Metier'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
# save data
    Save_(X_train, y_train, train_data_path_after_split)
    Save_(X_test, y_test, test_data_path_after_split)
    # Save_(X_val, y_val, val_data_path_after_split)
# split data to Numeric, Ordinal, and Nominal (Not_Ordinal)
    Ord_train, numeric_train = preprocess_py.split_(X_train)
    x_train = preprocess_py.preprocess(Ord_train,
                                       numeric_train, dataset_Typee=False)

    Ord_test, numeric_test = preprocess_py.split_(X_test)
    x_test = preprocess_py.preprocess(Ord_test,
                                      numeric_test, dataset_Typee=True)
# Tune the model
    best_score, best_params =Tune_Random_Forest(x_train, y_train)
    print(best_score)

    mod = RandomForestClassifier(bootstrap=best_params['bootstrap'], min_samples_split=best_params['min_samples_split'],n_estimators=best_params['n_estimators'])
    # DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = 8)

# fit the model & save it in Models directory + print the Metrics of the model (Precision + Recall)
    mod.fit(x_train, y_train)

    joblib.dump(mod, model_name)

    y_pred = mod.predict(x_test)
    
    result = compute_metrics(y_pred, y_test['Metier'].values)

    return result