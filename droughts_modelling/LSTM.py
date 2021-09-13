from droughts_modelling.data import DataFunctions
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from droughts_modelling.window_gen import WindowGenerator
from tensorflow.keras import layers, models
import tensorflow
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1_l2
from google.cloud import storage

class LSTM():
    
    def __init__(self, breg, kreg):
        self.train_data = DataFunctions().light_weekly_aggregate_train()
        self.breg = breg
        self.kreg = kreg
        #  self.train_data = train_data[train_data['fips_']<= 5035] #Currently set up for first c.100 fips, just drop this line to train fully
        self.features = self.train_data.drop(columns=['fips_','year_','week_num_','score_max']).columns
    
    #Data Scaling: Train and Test
    def robust(self):
        train_df = self.train_data.copy()
        for f in self.features:
            train_median = np.median(train_df[f])
            train_iqr = np.subtract(*np.percentile(train_df[f], [75, 25]))
            train_df[f] = train_df[f].map(lambda x: (x-train_median)/train_iqr)
        
        self.train_df_robust = train_df
        
    #One Hot Encoding: Train and Test
    def ohe(self):
        self.robust()
        train_df = self.train_df_robust.copy()
        
        ohe = OneHotEncoder(sparse = False)
        ohe.fit(train_df[['score_max']])
        
        scoremax_encoded_train = ohe.transform(train_df[['score_max']])
        for n in range(len(scoremax_encoded_train.T)):
            train_df[f"score_max_{n}"] = scoremax_encoded_train.T[n]
        train_df.drop(columns=['score_max','year_','week_num_'],inplace=True)    
        self.train_df_robust_ohe = train_df

    #Generating Windows: Train and Test    
    def window(self):
        self.ohe()
        train_df = self.train_df_robust_ohe
        
        def fip_splitter(df):
            df1 = df[df['fips_'] == 1001]
            df1 = df1.drop(columns='fips_')
            window = WindowGenerator(df1,input_width=6,label_width=6,shift=6,label_columns=[c for c in df.columns if 'score_max' in c]).make_dataset()
            for fips in set(df['fips_']):
                if fips > 1001:
                    fip_df = df[df['fips_'] == fips]
                    fip_window = WindowGenerator(fip_df.drop(columns=['fips_']),input_width=6,label_width=6,shift=6,label_columns=[c for c in df.columns if 'score_max' in c]).make_dataset()
                    window = window.concatenate(fip_window)
            return window
            
        self.train_metawindow = fip_splitter(train_df)

    #Model + evaluation
    def initialize_model(self):
        self.model = models.Sequential()
        self.model.add(layers.LSTM(128,return_sequences=True,activation='tanh',bias_regularizer=self.breg,kernel_regularizer=self.kreg))
        self.model.add(layers.LSTM(128,return_sequences=True,activation='tanh',bias_regularizer=self.breg,kernel_regularizer=self.kreg))
        self.model.add(layers.LSTM(64,return_sequences=True,activation='tanh',bias_regularizer=self.breg,kernel_regularizer=self.kreg))
        self.model.add(layers.LSTM(64,return_sequences=True,activation='tanh',bias_regularizer=self.breg,kernel_regularizer=self.kreg))
        self.model.add(layers.LSTM(64,return_sequences=False,activation='tanh',bias_regularizer=self.breg,kernel_regularizer=self.kreg))
        self.model.add(layers.Dense(6*6,kernel_initializer=tensorflow.initializers.zeros(),activation='softmax'))
        self.model.add(layers.Reshape([6,6]))
        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        
    def train_model(self):
        self.initialize_model()
        self.window()
        self.model.fit(self.train_metawindow,epochs=50,batch_size=32,callbacks=EarlyStopping(monitor='accuracy',patience=10,restore_best_weights=True),verbose=1)
        
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
        MODEL_VERSION = 'LSTM_1'
        
        client = storage.Client().bucket(BUCKET_NAME)
        storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{local_model_name}"
        blob = client.blob(storage_location)
        blob.upload_from_filename(local_model_name)
        

if __name__ == '__main__':
    my_test = LSTM(breg=l1_l2(l1=0.00, l2=0.00),kreg=l1_l2(l1=0, l2=0.02))
    my_test.train_model()
    my_test.train_evaluate_model()
    print('model trained')
    my_test.save_model_locally()
    print('saved locally')
    my_test.save_model_to_gcp()
    print('saved GCP')
    