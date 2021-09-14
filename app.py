import os
import streamlit as st
import tensorflow as tf
import pandas as pd 
import numpy as np
import datetime
from droughts_modelling.window_gen import WindowGenerator
from droughts_modelling.data import DataFunctions
from sklearn.preprocessing import OneHotEncoder
import base64

pages = ["Predict","About Project","Data Exploration"]
page = st.sidebar.radio("Navigate", options=pages)
st.title(page)

main_bg = "crack_cracked_earth_2.jpg"
main_bg_ext = "jpg"

side_bg = "crack_cracked_earth_2.jpg"
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)

file_path = os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
full_path_fips = os.path.join(file_path,'code','realGhostFoxx','droughts_modelling', 'raw_data', 'fips_dict.csv')
fips_dict = pd.read_csv(full_path_fips,index_col=[0])

if page == "Predict": 
    shift = 6
    inputs = 6
    
    date = st.date_input("Date")
    week = date.isocalendar()[1]
    year = date.isocalendar()[0]  
    
    one_count = st.radio('Choose one county?',options=["yes","no"])
    
    if one_count == "yes":
        fips_dict["UniqueCounty"] = fips_dict["COUNTYNAME"]+", "+fips_dict["STATE"]
        county_string = st.selectbox(label="County",               
        options=fips_dict["UniqueCounty"],
        key=fips_dict["fips"]) 
        
        county = county_string[:-4]
        state = county_string[-2:]
        fips = fips_dict[fips_dict["STATE"]==state][fips_dict[fips_dict["STATE"]==state]["COUNTYNAME"] == county]["fips"].iloc[0]
    
    else:
        fips = None   
        
    if (year <= 2018) | (year >= 2021):
        st.write("Pick different date") 
    
    else:                 
        if st.button('Predict'):
            predict_date = date+datetime.timedelta(weeks=shift)
            st.write(f'Future prediction date will be {predict_date}, which is {shift} weeks into the future.')
            test_df_robust_ohe = pd.read_csv("test_df_robust_ohe.csv",index_col=0)
                  
            if fips:                        
                fips_df = test_df_robust_ohe[test_df_robust_ohe["fips_"] == fips]
            else:
                fips_df = test_df_robust_ohe
            
            year_df = fips_df[fips_df["year_"] == year]
            week_df = year_df[year_df["week_num_"] == week]
            
            predict_df = pd.DataFrame()
            
            st.write(week_df)
            
            for index in week_df.index:
                if inputs > (year-2019)*52+week:
                    st.write(f"Pick fewer inputs or later date")
                    predict_df = 0
                else:
                    predict_df = predict_df.append(fips_df.loc[index-inputs+1:index+shift,:])
                    
            
            st.write(predict_df)                    
                
            predict_df.drop(columns=["fips_","year_"],inplace=True)
            wingen = WindowGenerator(predict_df,input_width=inputs,label_width=1,shift=shift,label_columns=["score_max_0","score_max_1","score_max_2","score_max_3","score_max_4","score_max_5"])
            predict_data = wingen.split_window
            predict_data = wingen.make_dataset()      
            
            model = tf.keras.models.load_model('trained_model_2021-09-11 13_01_35.778360.h5')
            
            prediction = model.predict(predict_data)
            
            if fips:
                likelihood = prediction[0][-1].max()
                st.write(prediction[0][-1])
                score = np.where(prediction[0][-1]==likelihood)
                likelihood = round(likelihood,4)
                
                st.write(f"""On {predict_date} in {county} it is predicted 
                that the drought classification will be {score[0][0]} with a """+str(likelihood)+f"% chance.")
            else:
                st.write(len(prediction))
                score_list = []
                for arr in prediction:
                    likelihood = arr[-1].max()
                    score = np.where(arr[-1]==likelihood)
                    score_list.append(score[0][0])
                st.write(score_list)
                
        else:
            pass
        
elif page == "About Project":
   st.write("The aim of this project was to build a Deep Learning model to predict future occurences of drought using meteorological time series data. The training data we used comprised of 3100 simultaneous sequences, each of 16 years, each belonging to a US county, simply stacked on top of eachother in a c.17,000,000 row CSV file.")
   
   st.write("The model used is a Long Short Term Memory (LSTM) model - a kind of recurrent neural network well suited to time series data. The model has been trained on countless data windows, configured in such a way as to allow it to predict as far as X weeks into the future, using X weeks of historical data to do so." )
   
   st.write("Currently the model is predicting with an estimated accuracy of X, a fair score considering the complexity of the task and the short period of time it was built in!")
   
   st.write("The dataset and the original kaggle post that inspired this project can be found here: https://www.kaggle.com/cdminix/us-drought-meteorological-data")
   
else:
    pass 




