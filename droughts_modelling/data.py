import os
import pandas as pd

def train_last_2_years():
    
    file_path = os.path.dirname(os.path.dirname(os.getcwd()))
    full_path = os.path.join(file_path, 'droughts_modelling', 'raw_data', 'train_timeseries.csv')
    
    big_df = pd.read_csv(full_path)
    big_df['date'] = pd.to_datetime(big_df['date'])
    
    temp_df = big_df[big_df['date'] >= '2015-01-01']
    
    return temp_df