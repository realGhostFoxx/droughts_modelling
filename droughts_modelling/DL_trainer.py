from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import models,layers
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from droughts_modelling.data import DataFunctions
from droughts_modelling.window_gen import WindowGenerator
import numpy as np
import joblib
from google.cloud import storage

class DeepLearning:
    
    def __init__(self):
        self.data = DataFunctions().light_weekly_aggregate_train()
        self.features = self.data.drop(columns=['fips_','year_','week_num_','score_max']).columns
        
    def robust(self):
        df = self.data.copy()
        for f in self.features:
            median = np.median(df[f])
            iqr = np.subtract(*np.percentile(df[f], [75, 25]))
            df[f] = df[f].map(lambda x: (x-median)/iqr)
        
        self.scaled_data = df
    
    def ohe(self):
        self.robust()
        df = self.scaled_data.copy()
        ohe = OneHotEncoder(sparse = False)
        ohe.fit(df[['score_max']])
        scoremax_encoded = ohe.transform(df[['score_max']])
        df["score_max_0"],df["score_max_1"],df['score_max_2'],df['score_max_3'],df['score_max_4'],df['score_max_5'] = scoremax_encoded.T 
        self.scaled_data_ohe = df.drop(columns=['score_max'])
           
    def preprocess(self):
        self.ohe()
        self.preprocessed_data = WindowGenerator(self.scaled_data_ohe,input_width=6,label_width=6,shift=1,label_columns=["score_max_0","score_max_1","score_max_2","score_max_3","score_max_4","score_max_5"]).make_dataset()
        
    def initialize_model(self):
        self.model = models.Sequential()
        self.model.add(layers.LSTM(32,return_sequences=True,activation='tanh'))
        self.model.add(layers.LSTM(32,return_sequences=True,activation='tanh'))
        self.model.add(layers.Dense(20,activation='relu'))
        self.model.add(layers.Dense(6,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        
    def train_model(self):
        self.initialize_model()
        self.preprocess()
        self.model.fit(self.preprocessed_data,epochs=1000,batch_size=32,verbose=1)
        
    #def create_pipe(self):
        #df = self.data
        #preproc = Pipeline([(RobustScaler(),df.drop(columns=['fips_','year_','week_num_','score_max']).columns)])
        #model = KerasClassifier(self.initialize_model,epochs=1000, batch_size=16, verbose=0)
        #pipe = Pipeline([(preproc,model)])
        #return pipe
    
        
        