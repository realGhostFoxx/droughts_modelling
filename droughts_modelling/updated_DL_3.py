from droughts_modelling.preprocess import Preprocess
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import layers, models
import tensorflow
import numpy as np

class DeepLearning3():
    
    def __init__(self):
        self.data = Preprocess().window()
    
    def initialize_model(self,breg,kreg):
        self.model = models.Sequential()
        self.model.add(layers.LSTM(128,return_sequences=True,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
        self.model.add(layers.LSTM(128,return_sequences=True,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
        self.model.add(layers.LSTM(64,return_sequences=True,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
        self.model.add(layers.LSTM(64,return_sequences=True,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
        self.model.add(layers.LSTM(64,return_sequences=False,activation='tanh',bias_regularizer=None,kernel_regularizer=None))
        self.model.add(layers.Dense(6*6,kernel_initializer=tensorflow.initializers.zeros(),activation='softmax'))
        self.model.add(layers.Reshape([6,6]))
        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        return self.model
        
    def train_model(self):
        self.initialize_model()
        self.model.fit(self.data,epochs=50,batch_size=32,callbacks=EarlyStopping(patience=10,restore_best_weights=True),verbose=0)
    
    #def save_model_to_gcp(self):
        # joblib.dump(local_model_name)
        #local_model_name = 'model.joblib'
        #joblib.dump(self.model, local_model_name)
        #print("saved model.joblib locally")
        #client = storage.Client().bucket(BUCKET_NAME)
        #storage_location = f"models/{MODEL_NAME}/{MODEL_VERSION}/{self.model}"
        #blob = client.blob(storage_location)
        #blob.upload_from_filename(local_model_name)
        #print('saved_gcp')
        
if __name__ == '__main__':
    my_test = DeepLearning3()
    my_test.train_evaluate_model()
        
        