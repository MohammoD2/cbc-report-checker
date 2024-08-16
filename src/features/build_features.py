import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_features(input_csv_path, output_dir):
    # Load cleaned data
    data = pd.read_csv(input_csv_path)
    
    # Separate features (X) and target variable (y)
    X = data.drop('Target', axis=1)
    y = data['Target']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize StandardScaler and fit to training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert numpy arrays back to DataFrames for saving
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)
    
    # Save the scaled datasets
    X_train_scaled_df.to_csv(f'{output_dir}/X_train_scaled.csv', index=False)
    X_test_scaled_df.to_csv(f'{output_dir}/X_test_scaled.csv', index=False)
    y_train.to_csv(f'{output_dir}/y_train.csv', index=False)
    y_test.to_csv(f'{output_dir}/y_test.csv', index=False)
    
    print(f'Saved scaled data to {output_dir}')

if __name__ == "__main__":
    input_csv_path = 'E:\Work files\cbc-report-checker\data\processed\cleaned_data.csv'  # Adjust the path as necessary
    output_dir = 'E:\Work files\cbc-report-checker\data\processed'  # Output directory where files will be saved
    
    # Call the function to process and save features
    build_features(input_csv_path, output_dir)
