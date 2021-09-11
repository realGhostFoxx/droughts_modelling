import streamlit as st
import tensorflow as tf
import pandas as pd 
import numpy as np
from droughts_modelling.window_gen import WindowGenerator
from droughts_modelling.data import DataFunctions
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model

pages = ["Predict","About Project","Data Exploration"]
page = st.sidebar.radio("Navigate", options=pages)
st.title(page)

if page == "Predict":
    
    fips_dict = pd.read_csv("raw_data/fips_dict.csv")
    fips_dict["UniqueCounty"] = fips_dict["COUNTYNAME"]+", "+fips_dict["STATE"]

    county_string = st.selectbox(label="County",
    options=fips_dict["UniqueCounty"],
    key=fips_dict["fips"]) 

    county = county_string[:-4]
    state = county_string[-2:]

    fips = fips_dict[fips_dict["STATE"]==state][fips_dict[fips_dict["STATE"]==state]["COUNTYNAME"] == county]["fips"].iloc[0]
    date = st.date_input("Date")
   
    week = date.isocalendar()[1]
    year = date.isocalendar()[0]        
    
    shift = 4
    inputs = 7
    
    data_func = DataFunctions(local=True)
    fips_dict = data_func.fips_dict
    test_data = data_func.light_weekly_aggregate_test()    
    train_data = data_func.light_weekly_aggregate_train()
    
    features = train_data.drop(columns=['fips_','year_','week_num_','score_max']).columns
    
    train_df = train_data.copy()
    test_df = test_data.copy()
    
    for f in features:
     train_median = np.median(train_df[f])
     train_iqr = np.subtract(*np.percentile(train_df[f], [75, 25]))
     train_df[f] = train_df[f].map(lambda x: (x-train_median)/train_iqr)
     test_df[f] = test_df[f].map(lambda x: (x-train_median)/train_iqr)
        
    test_df_robust = test_df        
                
    test_df = test_df_robust.copy()
    test_ohe = OneHotEncoder(sparse = False)
    test_ohe.fit(test_df[['score_max']]) 
    scoremax_encoded_test = test_ohe.transform(test_df[['score_max']])   
    test_df["score_max_0"],test_df["score_max_1"],test_df['score_max_2'],test_df['score_max_3'],test_df['score_max_4'],test_df['score_max_5'] = scoremax_encoded_test.T     
    test_df_robust_ohe = test_df.drop(columns=['score_max'])               

    fips_df = test_df_robust_ohe[test_data["fips_"] == fips]
    year_df = fips_df[fips_df["year_"] == year]
    week_df = year_df[year_df["week_num_"] == week]

    index = week_df.index[0]

    if inputs > (year-2019)*52+week:
        print(f"Pick fewer inputs or later date")
        predict_df = 0
    else:
        predict_df = fips_df.loc[index-inputs+1:index+shift,:]

    wingen = WindowGenerator(predict_df,input_width=inputs,label_width=1,shift=shift,label_columns=["score_max_0","score_max_1","score_max_2","score_max_3","score_max_4","score_max_5"])
    predict_data = wingen.split_window
    st.write(predict_data)
    predict_data = wingen.make_dataset()      
    st.write(predict_data)
    
    numpy_images = []
    numpy_labels = []

    for images, labels in predict_data.take(9):  # only take first element of dataset
        numpy_images = images.numpy()
        numpy_labels = labels.numpy()

    numpy_images    
                
    model = load_model('trained_model_2021-09-11 13_01_35.778360.h5')
    st.write(model.predict(predict_data))     
              
    if st.button('Predict'):
                
        trained_model = tf.keras.models.load_model('trained_model_2021-09-11 13_01_35.778360.h5')
        prediction = trained_model.predict(predict_data)
        
        st.write(prediction)
        
        score = "SCORE"
        description = "DESCRIPTION"
        likelihood = "LIKELIHOOD"
        
        st.write(f"""On the {date} in {county} it is predicted 
        that the drought classification will be {score} – 
        {description} with a {likelihood}% chance.  
        """)
    else:
        pass

elif page == "About project":
    
   pass 