from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras import models,layers
from sklearn.preprocessing import OneHotEncoder
from droughts_modelling.data import DataFunctions
from droughts_modelling.window_gen import WindowGenerator
import numpy as np

class DeepLearning2():
    
    def __init__(self):
        self.train_data = DataFunctions().light_weekly_aggregate_train()
        self.test_data = DataFunctions().light_weekly_aggregate_test()
        self.features = self.train_data.drop(columns=['fips_','year_','week_num_','score_max']).columns
    
    #Data Scaling: Train and Test
    def robust(self):
        train_df = self.train_data.copy()
        test_df = self.test_data.copy()
        for f in self.features:
            train_median = np.median(train_df[f])
            train_iqr = np.subtract(*np.percentile(train_df[f], [75, 25]))
            train_df[f] = train_df[f].map(lambda x: (x-train_median)/train_iqr)
            test_df[f] = test_df[f].map(lambda x: (x-train_median)/train_iqr)
            
        self.train_df_robust = train_df
        self.test_df_robust = test_df
    
    #Train data preprocessing
    def train_ohe(self):
        self.robust()
        df = self.train_df_robust.copy()
        ohe = OneHotEncoder(sparse = False)
        ohe.fit(df[['score_max']])
        scoremax_encoded = ohe.transform(df[['score_max']])
        df["score_max_0"],df["score_max_1"],df['score_max_2'],df['score_max_3'],df['score_max_4'],df['score_max_5'] = scoremax_encoded.T 
        self.train_df_robust_ohe = df.drop(columns=['score_max'])
        
    def train_window(self):
        self.train_ohe()
        self.train_windowed_data = WindowGenerator(self.train_df_robust_ohe,input_width=6,label_width=6,shift=1,label_columns=["score_max_0","score_max_1","score_max_2","score_max_3","score_max_4","score_max_5"]).make_dataset()
    
    #Test data preprocessing
    def test_ohe(self):
        self.robust()
        df = self.test_df_robust.copy()
        ohe = OneHotEncoder(sparse = False)
        ohe.fit(df[['score_max']])
        scoremax_encoded = ohe.transform(df[['score_max']])
        df["score_max_0"],df["score_max_1"],df['score_max_2'],df['score_max_3'],df['score_max_4'],df['score_max_5'] = scoremax_encoded.T 
        self.test_df_robust_ohe = df.drop(columns=['score_max']) 
    
    def test_window(self):
        self.test_ohe()
        self.test_windowed_data = WindowGenerator(self.test_df_robust_ohe,input_width=6,label_width=6,shift=1,label_columns=["score_max_0","score_max_1","score_max_2","score_max_3","score_max_4","score_max_5"]).make_dataset()
    
    #Model + evaluation
    def initialize_model(self):
        self.model = models.Sequential()
        self.model.add(layers.LSTM(32,return_sequences=True,activation='tanh'))
        self.model.add(layers.LSTM(32,return_sequences=True,activation='tanh'))
        self.model.add(layers.Dense(20,activation='relu'))
        self.model.add(layers.Dense(6,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        
    def train_model(self):
        self.initialize_model()
        self.train_window()
        self.model.fit(self.train_windowed_data,epochs=1,batch_size=32,verbose=1)
        
    def evaluate_model(self):
        self.train_model()
        self.test_window()
        self.model.evaluate(self.test_windowed_data,verbose=1)