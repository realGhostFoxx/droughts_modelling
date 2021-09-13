import os
import pandas as pd
import numpy as np
import datetime as dt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.tree import DecisionTreeClassifier
import ast

# BUCKET_NAME='drought-modelling-models'
# DATA_BUCKET_NAME = 'drought-modelling-datasets'
# BUCKET_TRAIN_DATA_PATH = 'data/train_timeseries.csv'
# BUCKET_VAL_DATA_PATH = 'data/validation_timeseries.csv'
# BUCKET_TEST_DATA_PATH = 'data/test_timeseries.csv'
# BUCKET_FIPS_PATH = 'data/fips_dict.csv'

class DataFunctions():
    
    def __init__(self, local=False):
        
        file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
        full_path_train = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'train_timeseries.csv')
        full_path_validate = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'validation_timeseries.csv')
        full_path_test = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'test_timeseries.csv')
        full_path_fips = os.path.join(file_path,'realGhostFoxx','droughts_modelling', 'raw_data', 'fips_dict.csv')
        
        BUCKET_NAME='drought-modelling-models'
        DATA_BUCKET_NAME = 'drought-modelling-datasets'
        BUCKET_TRAIN_DATA_PATH = 'data/train_timeseries.csv'
        BUCKET_MICRO_TRAIN_DATA_PATH = 'data/micro_train.csv'
        BUCKET_VAL_DATA_PATH = 'data/validation_timeseries.csv'
        BUCKET_TEST_DATA_PATH = 'data/test_timeseries.csv'
        BUCKET_FIPS_PATH = 'data/fips_dict.csv'
        
        if local:
            self.train_data = pd.read_csv(full_path_train)[2:]
            self.validation_data = pd.read_csv(full_path_validate)[1:]
            self.test_data = pd.read_csv(full_path_test)[6:]
            self.fips_dict = pd.read_csv(full_path_fips,index_col=[0])
        else:
            self.train_data = pd.read_csv(f"gs://{DATA_BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
            # self.train_data = pd.read_csv(f"gs://{DATA_BUCKET_NAME}/{BUCKET_TRAIN_DATA_PATH}")
            self.validation_data = pd.read_csv(f"gs://{DATA_BUCKET_NAME}/{BUCKET_VAL_DATA_PATH}")
            self.test_data = pd.read_csv(f"gs://{DATA_BUCKET_NAME}/{BUCKET_TEST_DATA_PATH}")
            self.fips_dict = pd.read_csv(f"gs://{DATA_BUCKET_NAME}/{BUCKET_FIPS_PATH}")
 
    def light_weekly_aggregate_train(self, scope='all'):
        
        if scope != 'all':
            df = self.train_data[self.train_data['fips'].isin([1001, 1003, 1005, 1015])]
        else:
            df = self.train_data
   
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['week_num'] = pd.to_datetime(df['date']).dt.week
        df['score_day'] = df['score'].apply(lambda x: 'yes' if pd.notnull(x) == True else '')

        aggregated_data_train = df.groupby(['fips','year','week_num']).agg(
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

        aggregated_data_train.columns = ['_'.join(col) for col in aggregated_data_train.columns.values]
        aggregated_data_train['score_max'] = aggregated_data_train['score_max'].map(lambda x: np.round(x))
        
        aggregated_data_train['sin_week'] = np.sin(2*np.pi*aggregated_data_train['week_num_']/52)
        aggregated_data_train['cos_week'] = np.cos(2*np.pi*aggregated_data_train['week_num_']/52)
        
        fips_dict = self.fips_dict.drop(columns=['COUNTYNAME',"STATE",'geom']).rename(columns={'fips':'fips_'})
        fips_dict["lat_long"] = fips_dict["lat_long"].transform(lambda x: ast.literal_eval(x))
        fips_dict["lat_rad"] = pd.DataFrame(fips_dict["lat_long"].tolist())[0].map(lambda x: (x * np.pi)/180)
        fips_dict["long_rad"] = pd.DataFrame(fips_dict["lat_long"].tolist())[1].map(lambda x: (x * np.pi)/180)
        fips_dict.drop(columns=["lat_long"],inplace=True)
        
        aggregated_data_train = pd.merge(aggregated_data_train,fips_dict, on=["fips_"], how="inner")
        aggregated_data_train = aggregated_data_train[['fips_','year_','week_num_','sin_week','cos_week', 'PRECTOT_mean', 'PS_mean', 'QV2M_mean',
       'T2M_mean', 'T2MDEW_mean', 'T2MWET_mean', 'T2M_MAX_mean',
       'T2M_MIN_mean', 'T2M_RANGE_mean', 'TS_mean', 'WS10M_mean',
       'WS10M_MAX_mean', 'WS10M_MIN_mean', 'WS10M_RANGE_mean', 'WS50M_mean',
       'WS50M_MAX_mean', 'WS50M_MIN_mean', 'WS50M_RANGE_mean','lat_rad', 'long_rad','score_max']]

        return aggregated_data_train.dropna()
    
    
    def light_weekly_aggregate_validate(self, scope='all'):
        
        if scope != 'all':
            df = self.train_data[self.train_data['fips'].isin([1001, 1003, 1005, 1015])]
        else:
            df = self.validation_data
   
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['week_num'] = pd.to_datetime(df['date']).dt.isocalendar().week

        df['score_day'] = df['score'].apply(lambda x: 'yes' if pd.notnull(x) == True else '')

        aggregated_data_validate = df.groupby(['fips','year','week_num']).agg(
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

        aggregated_data_validate.columns = ['_'.join(col) for col in aggregated_data_validate.columns.values]
        aggregated_data_validate['score_max'] = aggregated_data_validate['score_max'].map(lambda x: np.round(x))
       
        aggregated_data_validate['sin_week'] = np.sin(2*np.pi*aggregated_data_validate['week_num_']/52)
        aggregated_data_validate['cos_week'] = np.cos(2*np.pi*aggregated_data_validate['week_num_']/52)
    
        fips_dict = self.fips_dict.drop(columns=['COUNTYNAME','STATE','geom']).rename(columns={'fips':'fips_'})
        fips_dict["lat_long"] = fips_dict["lat_long"].transform(lambda x: ast.literal_eval(x))
        fips_dict["lat_rad"] = pd.DataFrame(fips_dict["lat_long"].tolist())[0].map(lambda x: (x * np.pi)/180)
        fips_dict["long_rad"] = pd.DataFrame(fips_dict["lat_long"].tolist())[1].map(lambda x: (x * np.pi)/180)
        fips_dict.drop(columns=["lat_long"],inplace=True)
        
        aggregated_data_validate = pd.merge(aggregated_data_validate,fips_dict, on=["fips_"], how="inner")
        aggregated_data_validate = aggregated_data_validate[['fips_','year_','week_num_','sin_week','cos_week', 'PRECTOT_mean', 'PS_mean', 'QV2M_mean',
       'T2M_mean', 'T2MDEW_mean', 'T2MWET_mean', 'T2M_MAX_mean',
       'T2M_MIN_mean', 'T2M_RANGE_mean', 'TS_mean', 'WS10M_mean',
       'WS10M_MAX_mean', 'WS10M_MIN_mean', 'WS10M_RANGE_mean', 'WS50M_mean',
       'WS50M_MAX_mean', 'WS50M_MIN_mean', 'WS50M_RANGE_mean','lat_rad','long_rad','score_max']]

        return aggregated_data_validate.dropna()

    def light_weekly_aggregate_test(self, scope='all'):
        
        if scope != 'all':
            df = self.train_data[self.train_data['fips'].isin([1001, 1003, 1005, 1015])]
        else:
            df = self.test_data
   
        df['year'] = pd.to_datetime(df['date']).dt.year
        df['week_num'] = pd.to_datetime(df['date']).dt.isocalendar().week

        df['score_day'] = df['score'].apply(lambda x: 'yes' if pd.notnull(x) == True else '')

        aggregated_data_test = df.groupby(['fips','year','week_num']).agg(
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

        aggregated_data_test.columns = ['_'.join(col) for col in aggregated_data_test.columns.values]
        aggregated_data_test['score_max'] = aggregated_data_test['score_max'].map(lambda x: np.round(x))
        
        aggregated_data_test['sin_week'] = np.sin(2*np.pi*aggregated_data_test['week_num_']/52)
        aggregated_data_test['cos_week'] = np.cos(2*np.pi*aggregated_data_test['week_num_']/52)
                
        fips_dict = self.fips_dict.drop(columns=['COUNTYNAME',"STATE",'geom']).rename(columns={'fips':'fips_'})
        fips_dict["lat_long"] = fips_dict["lat_long"].transform(lambda x: ast.literal_eval(x))
        fips_dict["lat_rad"] = pd.DataFrame(fips_dict["lat_long"].tolist())[0].map(lambda x: (x * np.pi)/180)
        fips_dict["long_rad"] = pd.DataFrame(fips_dict["lat_long"].tolist())[1].map(lambda x: (x * np.pi)/180)
        fips_dict.drop(columns=["lat_long"],inplace=True)
        
        aggregated_data_test = pd.merge(aggregated_data_test,fips_dict, on=["fips_"], how="inner")
        aggregated_data_test = aggregated_data_test[['fips_','year_','week_num_','sin_week','cos_week','PRECTOT_mean', 'PS_mean', 'QV2M_mean',
       'T2M_mean', 'T2MDEW_mean', 'T2MWET_mean', 'T2M_MAX_mean',
       'T2M_MIN_mean', 'T2M_RANGE_mean', 'TS_mean', 'WS10M_mean',
       'WS10M_MAX_mean', 'WS10M_MIN_mean', 'WS10M_RANGE_mean', 'WS50M_mean',
       'WS50M_MAX_mean', 'WS50M_MIN_mean', 'WS50M_RANGE_mean','lat_rad', 'long_rad','score_max']]

        return aggregated_data_test.dropna()
   
    def k_best_features(self):
        df = self.light_weekly_aggregate_train(scope='all')
    
        y = round(df['score_max'])
        X = df.drop(columns=['fips_','week_num_','score_max'])
        
        k_best_f = SelectKBest(f_classif, k=10).fit(X, y)
        df_scores = pd.DataFrame({'features': X.columns, 'ANOVA F-value': k_best_f.scores_, 'pValue': k_best_f.pvalues_ })

        return df_scores.sort_values('ANOVA F-value', ascending=False).reset_index()
    
    def tree_feature_importance(self):
        df = self.light_weekly_aggregate_train(scope='all')
    
        y = round(df['score_max'])
        X = df.drop(columns=['year_', 'fips_','week_num_','score_max'])
        
        tree_clf = DecisionTreeClassifier(max_depth=6, random_state=2)
        tree_clf.fit(X,y)

        features_df = pd.DataFrame({'features': X.columns, 'Feature Importance': tree_clf.feature_importances_})\
            .sort_values('Feature Importance', ascending=False).iloc[:20]
            
        return [i for i in features_df.features][:5]
    