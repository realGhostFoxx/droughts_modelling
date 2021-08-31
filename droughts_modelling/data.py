import os
import pandas as pd
import geopandas as gpd
import numpy as np
import datetime as dt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier

class DataFunctions:
    
    def __init__(self):
        file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
        full_path = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'train_timeseries.csv')
        fips_path = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'fips_dict.csv')
        self.data = pd.read_csv(full_path)[2:]
        self.fips_dict = pd.read_csv(fips_path)[2:]
    
    def train_last_2_years(self):
        df = self.data
        df['date'] = pd.to_datetime(df['date'])
        temp_df = df[df['date'] >= '2015-01-01']
    
        return temp_df

    def weekly_aggregate(self):
        df = self.data
        
        #first create new features: month, weekday, weeknum
        df['week_num'] = pd.to_datetime(df['date']).dt.isocalendar().week
    
        #then encode the score as a new feature - not sure if we'll need it 
        df['score_day'] = df['score'].apply(lambda x: 'yes' if pd.notnull(x) == True else '')

        #then start aggregating by fips, month, week_num
        aggregated_data_train = df.groupby(['fips','week_num']).agg(
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
                                         'score': 'max'}).reset_index().sort_values(['fips','week_num'])

        #finally, remove the multiindex from aggregated data_train so it looks neat and has flat column name structure
        #Then round scores to nearest integer
        aggregated_data_train.columns = ['_'.join(col) for col in aggregated_data_train.columns.values]
        aggregated_data_train['score_max'] = aggregated_data_train['score_max'].map(lambda x: np.round(x))
        
        return aggregated_data_train.dropna()
    
    def light_weekly_aggregate(self):
        df = self.data

        fips_dict = self.fips_dict

        #first create new features: year, month, weekday, weeknum

        #first create new features: month, weekday, weeknum

        df['week_num'] = pd.to_datetime(df['date']).dt.isocalendar().week

        #then encode the score as a new feature - not sure if we'll need it 
        df['score_day'] = df['score'].apply(lambda x: 'yes' if pd.notnull(x) == True else '')

        #then start aggregating by fips, month, week_num
        aggregated_data_train = df.groupby(['fips','week_num']).agg(
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
                                         'score': 'max'}).reset_index().sort_values(['fips','week_num'])

        #finally, remove the multiindex from aggregated data_train so it looks neat and has flat column name structure
        #Then round scores to nearest integer
        aggregated_data_train.columns = ['_'.join(col) for col in aggregated_data_train.columns.values]
        aggregated_data_train['score_max'] = aggregated_data_train['score_max'].map(lambda x: np.round(x))

        fips_dict["lat_long"] = fips_dict["lat_long"].transform(lambda x: ast.literal_eval(x))
        fips_dict["lat"] = pd.DataFrame(fips_dict["lat_long"].tolist())[0]
        fips_dict["long"] = pd.DataFrame(fips_dict["lat_long"].tolist())[1]
        fips_dict.drop(columns=["lat_long"],inplace=True)
        
        aggregated_data_train = pd.merge(aggregated_data_train,fips_dict, on=["fips"], how="inner")
        return aggregated_data_train.dropna()
    
    def k_best_features(self):
        df = self.light_weekly_aggregate()
    
        y = round(df['score_max'])
        X = df.drop(columns=['fips_','week_num_','score_max'])
        
        k_best_f = SelectKBest(f_classif, k=10).fit(X, y)
        df_scores = pd.DataFrame({'features': X.columns, 'ANOVA F-value': k_best_f.scores_, 'pValue': k_best_f.pvalues_ })

        return df_scores.sort_values('ANOVA F-value', ascending=False).reset_index()
    
    def tree_feature_importance(self):
        df = self.light_weekly_aggregate()
    
        y = round(df['score_max'])
        X = df.drop(columns=['fips_','week_num_','score_max'])
        
        tree_clf = DecisionTreeClassifier(max_depth=6, random_state=2)
        tree_clf.fit(X,y)

        return pd.DataFrame({'features': X.columns, 'Feature Importance': tree_clf.feature_importances_})\
            .sort_values('Feature Importance', ascending=False).iloc[:20]
            
    def return_lagged_function(self, weeks_back=5):
        
        df = self.light_weekly_aggregate()
        
        top_features = ['T2M_RANGE_mean', 'PS_mean', 'T2M_MAX_mean', 'TS_mean', 
                        'T2MDEW_mean', 'QV2M_mean', 'WS10M_MAX_mean', 'PRECTOT_mean']
        
        all_features = [i for i in df.columns if i in top_features or i in ['fips_', 'year_', 'week_num_']]
        
        df_processed = df[all_features]
        
        for e in top_features:
            for i in range(1, weeks_back):
                df_processed[f'{e} - {i}'] = df_processed.groupby(['fips_'])[f'{e}'].shift(i)
                
    
       return df_processed 

