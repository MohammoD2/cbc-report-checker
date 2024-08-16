import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def plot_distribution(data, column):
    plt.figure(figsize=(8, 4))
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
    plt.xlabel(column)
    plt.ylabel('Frequency')
    plt.show()

def save_visualization(plt, filename):
    plt.savefig(filename)

if __name__ == "__main__":
    data = pd.read_csv('E:\Work files\cbc-report-checker\data\processed\cleaned_data.csv')
    
    columns_to_plot = ['Hemoglobin', 'WBC', 'Neutrophils', 'Lymphocytes', 'PLT']
    for column in columns_to_plot:
        plot_distribution(data, column)
