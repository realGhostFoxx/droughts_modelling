import os
import pandas as pd
import numpy as np
import datetime as dt

class DataFunctions:
    
    def train_last_2_years(self):
    
        file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
        full_path = os.path.join(file_path,'code','realGhostFoxx','droughts_modelling', 'raw_data', 'train_timeseries.csv')
    
        big_df = pd.read_csv(full_path)
        big_df['date'] = pd.to_datetime(big_df['date'])
    
        temp_df = big_df[big_df['date'] >= '2015-01-01']
    
        return temp_df


    def weekly_aggregate(self):
    
        file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
        full_path = os.path.join(file_path,'code','realGhostFoxx','droughts_modelling', 'raw_data', 'train_timeseries.csv')
        data = pd.read_csv(full_path)[2:]
    
        #first create new features: year, month, weekday, weeknum
        data['week_num'] = pd.to_datetime(data['date']).dt.isocalendar().week
        data['weekday'] = pd.to_datetime(data['date']).dt.weekday+1
        data['month'] = pd.to_datetime(data['date']).dt.month_name()
        data['year'] = pd.to_datetime(data['date']).dt.isocalendar().year

        #then encode the score as a new feature - not sure if we'll need it 
        data['score_day'] = data['score'].apply(lambda x: 'yes' if pd.notnull(x) == True else '')

        #then start aggregating by fips, year, month, week_num
        aggregated_data_train = data.groupby(['fips', 'year', 
                                        'month', 'week_num']).agg(
                                        {'PRECTOT': ['min', 'mean', 'std'],
                                        'PS': ['min', 'mean', 'std'],
                                        'QV2M': ['min', 'mean', 'std'],
                                        'T2M': ['min', 'mean', 'std'],
                                        'T2MDEW': ['min', 'mean', 'std'],
                                        'T2MWET': ['min', 'mean', 'std'],
                                        'T2M_MAX': ['min', 'mean', 'std'],
                                        'T2M_MIN': ['min', 'mean', 'std'],
                                        'T2M_RANGE': ['min', 'mean', 'std'],
                                         'TS': ['min', 'mean', 'std'],
                                         'WS10M': ['min', 'mean', 'std'],
                                         'WS10M_MAX': ['min', 'mean', 'std'],
                                         'WS10M_MIN': ['min', 'mean', 'std'],
                                         'WS10M_RANGE': ['min', 'mean', 'std'],
                                         'WS50M': ['min', 'mean', 'std'],
                                         'WS50M_MAX': ['min', 'mean', 'std'],
                                         'WS50M_MIN': ['min', 'mean', 'std'],
                                         'WS50M_RANGE': ['min', 'mean', 'std'],
                                         'score': 'max'}).reset_index().sort_values(['fips','year','week_num'])

        #finally, remove the multiindex from aggregated data_train so it looks neat and has flat column name structure
        #Then round scores to nearest integer
        aggregated_data_train.columns = ['_'.join(col) for col in aggregated_data_train.columns.values]
        aggregated_data_train['score_max'] = aggregated_data_train['score_max'].map(lambda x: np.round(x))
        return aggregated_data_train

