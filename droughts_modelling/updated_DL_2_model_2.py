from sklearn.preprocessing import OneHotEncoder
from tensorflow import keras
from tensorflow.keras import models,layers
from sklearn.preprocessing import OneHotEncoder
from droughts_modelling.data import DataFunctions
from droughts_modelling.window_gen import WindowGenerator
from datetime import datetime
import numpy as np
import tensorflow as tf
import pandas as pd
from google.cloud import storage

#This is the model that takes the correct data input

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
    
    #One Hot Encoding: Train and Test
    def ohe(self):
        self.robust()
        train_df = self.train_df_robust.copy()
        test_df = self.test_df_robust.copy()
        
        train_ohe = OneHotEncoder(sparse = False)
        test_ohe = OneHotEncoder(sparse = False)
        
        train_ohe.fit(train_df[['score_max']])
        test_ohe.fit(test_df[['score_max']])
        
        scoremax_encoded_train = train_ohe.transform(train_df[['score_max']])
        scoremax_encoded_test = test_ohe.transform(test_df[['score_max']])
        
        train_df["score_max_0"],train_df["score_max_1"],train_df['score_max_2'],train_df['score_max_3'],train_df['score_max_4'],train_df['score_max_5'] = scoremax_encoded_train.T 
        test_df["score_max_0"],test_df["score_max_1"],test_df['score_max_2'],test_df['score_max_3'],test_df['score_max_4'],test_df['score_max_5'] = scoremax_encoded_test.T 
        
        self.train_df_robust_ohe = train_df.drop(columns=['score_max'])
        self.test_df_robust_ohe = test_df.drop(columns=['score_max'])
        
    #Generating Windows: Train and Test    
    def window(self):
        self.ohe()
        train_df = self.train_df_robust_ohe
        test_df = self.test_df_robust_ohe
        
        def fip_splitter(df):
            window = WindowGenerator(df[df['fips_'] == 1001],input_width=6,label_width=6,shift=1,label_columns=["score_max_0","score_max_1","score_max_2","score_max_3","score_max_4","score_max_5"]).make_dataset()
            for fips in set(df['fips_']):
                if fips != 1001:
                    fip_df = df[df['fips_'] == fips]
                    fip_window = WindowGenerator(fip_df,input_width=6,label_width=6,shift=1,label_columns=["score_max_0","score_max_1","score_max_2","score_max_3","score_max_4","score_max_5"]).make_dataset()
                    window = window.concatenate(fip_window)
            return window
            
        self.train_metawindow = fip_splitter(train_df)
        self.test_metawindow = fip_splitter(test_df)

    #Model + evaluation
    def initialize_model(self,breg=tf.keras.regularizers.l1_l2(l1=0, l2=0.01), kreg=tf.keras.regularizers.l1_l2(l1=0, l2=0.01)):
            self.model = models.Sequential()
            self.model.add(layers.LSTM(128,return_sequences=True,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
            self.model.add(layers.LSTM(128,return_sequences=True,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
            self.model.add(layers.LSTM(64,return_sequences=True,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
            self.model.add(layers.LSTM(64,return_sequences=True,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
            self.model.add(layers.LSTM(64,return_sequences=False,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
            self.model.add(layers.Dense(6*6,kernel_initializer=tf.initializers.zeros(),activation='softmax'))
            self.model.add(layers.Reshape([6,6]))
            self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        
    def train_evaluate_model(self):
        self.initialize_model()
        self.window()
        self.model.fit(self.train_metawindow,epochs=1,batch_size=32,verbose=0)
        # self.save_model_to_gcp()
        # self.model.evaluate(self.test_metawindow,verbose=0)
        # return self.model
    
    def save_model_locally(self):
        self.model.save('model.h5')
        
    def save_model_to_gcp(self):
        
        BUCKET_NAME='drought-modelling-datasets'
        # BUCKET_TRAIN_DATA_PATH = 'data/train_timeseries.csv'
        # BUCKET_MICRO_TRAIN_DATA_PATH = 'data/micro_train.csv'
        # BUCKET_VAL_DATA_PATH = 'data/validation_timeseries.csv'
        # BUCKET_TEST_DATA_PATH = 'data/test_timeseries.csv'
        # BUCKET_FIPS_PATH = 'data/fips_dict.csv'
        
        local_model_name = 'model.h5'
        
        MODEL_NAME = 'model_trial'
        MODEL_VERSION = '4_new'
        
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)

if __name__ == '__main__':
    my_test = DeepLearning2()
    print('class instantiated')
    my_test.train_evaluate_model()
    print('model trained')
    my_test.save_model_locally()
    print('saved locally')
    my_test.save_model_to_gcp()
    print('saved GCP')