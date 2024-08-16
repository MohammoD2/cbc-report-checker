import pandas as pd

def load_data(file_path):
    data = pd.read_csv(file_path)
    return data

def clean_data(data):
    data = data.drop(columns=['PLT', 'WBC', 'ID'])
    data['PLT'] = data['PLT T']
    data['WBC'] = data['WBC T']
    data = data.drop(columns=['PLT T', 'WBC T'])
    return data

def save_clean_data(data, output_file):
    data.to_csv(output_file, index=False)

if __name__ == "__main__":
    data = load_data(r'E:\Work files\cbc-report-checker\data\raw\Raw_Data.csv')
    clean_data = clean_data(data)
    save_clean_data(clean_data, 'E:\Work files\cbc-report-checker\data\processed\cleaned_data.csv')
