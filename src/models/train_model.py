from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Random Forest Accuracy =", accuracy)
    return accuracy

def save_model(model, file_name):
    with open(file_name, 'wb') as f:
        pickle.dump(model, f)

if __name__ == "__main__":
    import pandas as pd
    X_train = pd.read_csv(r'E:\Work files\cbc-report-checker\data\processed\X_train_scaled.csv')
    y_train = pd.read_csv(r'E:\Work files\cbc-report-checker\data\processed\y_train.csv')
    X_test = pd.read_csv(r'E:\Work files\cbc-report-checker\data\processed\X_test_scaled.csv')
    y_test = pd.read_csv(r'E:\Work files\cbc-report-checker\data\processed\y_test.csv')
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    save_model(model, r'E:\Work files\cbc-report-checker\models\model.pkl')
