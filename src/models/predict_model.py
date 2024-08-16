import pickle
import pandas as pd

def load_model(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model

def load_scaler(scaler_path):
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    return scaler

def predict(data, model, scaler):
    data_scaled = scaler.transform(data)
    prediction = model.predict(data_scaled)
    return prediction

if __name__ == "__main__":
    model = load_model(r'E:\Work files\cbc-report-checker\models\model.pkl')
    scaler = load_scaler(r'E:\Work files\cbc-report-checker\models\scaler.pkl')
    
    # Assume user_input.csv is the file containing the features of a new sample
    new_data = pd.read_csv('user_input.csv')
    prediction = predict(new_data, model, scaler)
    print("Prediction:", prediction)
