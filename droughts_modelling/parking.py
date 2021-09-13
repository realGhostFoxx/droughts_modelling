    def weekly_aggregate(self):
        df = self.train_data
        
        #first create new features: month, weekday, weeknum
        df['year'] = pd.to_datetime(df['date']).dt.isocalendar().year
        df['week_num'] = pd.to_datetime(df['date']).dt.isocalendar().week
        
        #then encode the score as a new feature - not sure if we'll need it 
        df['score_day'] = df['score'].apply(lambda x: 'yes' if pd.notnull(x) == True else '')

        #then start aggregating by fips, month, week_num
        aggregated_data_train = df.groupby(['fips','year','week_num']).agg(
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
                 
    def train_last_2_years(self):
        df = self.train_data
        df['date'] = pd.to_datetime(df['date'])
        temp_df = df[df['date'] >= '2015-01-01']
    
        return temp_df
    
    def return_lagged_function(self, weeks_back=5):
        
        df = self.light_weekly_aggregate_train(scope='all')
        
        top_features = ['T2M_RANGE_mean', 'PS_mean', 'T2M_MAX_mean', 'TS_mean', 
                        'T2MDEW_mean', 'QV2M_mean', 'WS10M_MAX_mean', 'PRECTOT_mean']
        
        all_features = [i for i in df.columns if i in top_features or i in ['fips_', 'year_', 'week_num_']]
        
        df_processed = df[all_features]
        
        for e in top_features:
            for i in range(1, weeks_back):
                df_processed[f'{e} - {i}'] = df_processed.groupby(['fips_'])[f'{e}'].shift(i)
                
        return df_processed