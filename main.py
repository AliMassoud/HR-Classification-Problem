import pandas as pd
from src.ML_code.train import build_model
from src.ML_code.inference import make_predictions

# test the training part
def train_model():
    train_dt_path = 'Data/dataset_train_test.csv'
    training_data_df = pd.read_csv(train_dt_path, sep=';')
    model_performance_dict = build_model(training_data_df)
    print(model_performance_dict)

# test the predicting part
def predict_():
    test_dt_path = "Data/dataset_predict.csv"
    user_data_df = pd.read_csv(test_dt_path, sep=';')
    predictions = make_predictions(user_data_df)
    print(predictions)

# train_model()
predict_()