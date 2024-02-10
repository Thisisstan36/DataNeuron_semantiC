import pandas as pd

def preprocess_data(data_path):
    data = pd.read_csv(data_path)
    # Preprocess your data here
    return data