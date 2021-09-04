from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier

from droughts_modelling.data import DataFunctions

class HistGradientBooster:
    
    def __init__(self):
        self.data = DataFunctions().light_weekly_aggregate_train()
        
    def fit_train(self):
        df = self.data.copy()
        
        smol_df_train = df[df['year_'] < 2012]
        smol_df_test = df[df['year_'] >= 2013]
        
        train_X = smol_df_train.drop(columns=['year_', 'score_max'])
        train_y = smol_df_train['score_max']
        test_X = smol_df_test.drop(columns=['year_', 'score_max'])
        test_y = smol_df_test['score_max']
        
        model_1 = HistGradientBoostingClassifier(loss='categorical_crossentropy',
                                        l2_regularization=2.0).fit(train_X, train_y)
        
        return model_1
        