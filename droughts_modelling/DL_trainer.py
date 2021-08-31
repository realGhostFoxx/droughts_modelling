from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras import models,layers
from sklearn.pipeline import Pipeline
from droughts_modelling.data import DataFunctions
from droughts_modelling.window_gen import WindowGenerator
import numpy as np
import joblib
from google.cloud import storage

class DeepLearning:
    
    def __init__(self):
        self.data = DataFunctions().light_weekly_aggregate()
        self.features = self.data.drop(columns=['fips_','week_num_','score_max']).columns
        
    def robust(self):
        df = self.data.copy()
        for f in self.features:
            median = np.median(df[f])
            iqr = np.subtract(*np.percentile(df[f], [75, 25]))
            df[f] = df[f].map(lambda x: (x-median)/iqr)
        
        self.scaled_data = df
            
    def preprocess(self):
        self.robust()
        self.preprocessed_data = WindowGenerator(self.scaled_data,input_width=6, label_width=1, shift=1,label_columns=['score_max']).make_dataset()
        
    def initialize_model(self):
        self.model = models.Sequential()
        self.model.add(layers.LSTM(20))
        self.model.add(layers.Dense(1,activation='softmax'))
        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
    
    def train_model(self):
        self.initialize_model()
        self.preprocess()
        self.model.fit(self.preprocessed_data,epochs=1,batch_size=16,verbose=1)
        
    #def create_pipe(self):
        #df = self.data
        #preproc = Pipeline([(RobustScaler(),df.drop(columns=['fips_','year_','week_num_','score_max']).columns)])
        #model = KerasClassifier(self.initialize_model,epochs=1000, batch_size=16, verbose=0)
        #pipe = Pipeline([(preproc,model)])
        #return pipe
    
        
        