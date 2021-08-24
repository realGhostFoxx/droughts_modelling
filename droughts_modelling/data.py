import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

class DataFunctions:
    
    def train_last_2_years(self):
    
        file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
        full_path = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'train_timeseries.csv')
    
        big_df = pd.read_csv(full_path)
        big_df['date'] = pd.to_datetime(big_df['date'])
    
        temp_df = big_df[big_df['date'] >= '2015-01-01']
    
        return temp_df

    def weekly_aggregate(self):
    
        file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
        full_path = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'train_timeseries.csv')
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
                                        'week_num']).agg(
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
        
        return aggregated_data_train.dropna()
    
    def light_weekly_aggregate(self):
    
        file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
        full_path = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'train_timeseries.csv')
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
                                        'week_num']).agg(
                                        {'PRECTOT': ['mean'],
                                        'PS': ['mean'],
                                        'QV2M': ['mean'],
                                        'T2M': ['mean'],
                                        'T2MDEW': ['mean'],
                                        'T2MWET': ['mean'],
                                        'T2M_MAX': ['mean'],
                                        'T2M_MIN': ['mean'],
                                        'T2M_RANGE': ['mean'],
                                         'TS': ['mean'],
                                         'WS10M': ['mean'],
                                         'WS10M_MAX': ['mean'],
                                         'WS10M_MIN': ['mean'],
                                         'WS10M_RANGE': ['mean'],
                                         'WS50M': ['mean'],
                                         'WS50M_MAX': ['mean'],
                                         'WS50M_MIN': ['mean'],
                                         'WS50M_RANGE': ['mean'],
                                         'score': 'max'}).reset_index().sort_values(['fips','year','week_num'])

        #finally, remove the multiindex from aggregated data_train so it looks neat and has flat column name structure
        #Then round scores to nearest integer
        aggregated_data_train.columns = ['_'.join(col) for col in aggregated_data_train.columns.values]
        aggregated_data_train['score_max'] = aggregated_data_train['score_max'].map(lambda x: np.round(x))

        
        return aggregated_data_train.dropna()
    
    def k_best_features(self):
        df = light_weekly_aggregate()
    
        y = round(df['score_max'])
        X = df.drop(columns=['fips_', 'year_', 'week_num_', 'score_max'])
        
        k_best_f = SelectKBest(f_classif, k=10).fit(X, y)
        df_scores = pd.DataFrame({'features': X.columns, 'ANOVA F-value': k_best_f.scores_, 'pValue': k_best_f.pvalues_ })

        return df_scores.sort_values('ANOVA F-value', ascending=False).reset_index()
    
    def tree_feature_importance(self):
        df = light_weekly_aggregate()
    
        y = round(df['score_max'])
        X = df.drop(columns=['fips_', 'year_', 'week_num_', 'score_max'])
        
        tree_clf = DecisionTreeClassifier(max_depth=6, random_state=2)
        tree_clf.fit(X,y)

        return pd.DataFrame({'features': X.columns, 'Feature Importance': tree_clf.feature_importances_})\
            .sort_values('Feature Importance', ascending=False).iloc[:20]
      
      return aggregated_data_train.dropna()


