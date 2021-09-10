import streamlit as st
import pandas as pd 

st.write('Drought Prediction')
fips_dict = pd.read_csv("raw_data/fips_dict.csv")
fips_dict["UniqueCounty"] = fips_dict["COUNTYNAME"]+", "+fips_dict["STATE"]

a = st.selectbox(label="County",
options=fips_dict["UniqueCounty"],
key=fips_dict["fips"]) 

county = a[:-4]
state = a[-2:]

st.write(fips_dict[fips_dict["STATE"]==state][fips_dict[fips_dict["STATE"]==state]["COUNTYNAME"] == county]["fips"])

fips = fips_dict[fips_dict["STATE"]==state][fips_dict[fips_dict["STATE"]==state]["COUNTYNAME"] == county]["fips"].iloc[0]

st.write(fips)

date = st.date_input("Date")
st.write(date)



if st.button('Predict'):

    #score, likelihood = Call DL_predict
    score = "SCORE"
    description = "DESCRIPTION"
    likelihood = "LIKELIHOOD"
    
    st.write(f"""On the {date} in {county} it is predicted 
    that the drought classification will be {score} – 
    {description} with a {likelihood}% chance.  
    """)
else:
    0