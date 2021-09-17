FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt

COPY droughts_modelling /droughts_modelling
COPY app.py /app.py
COPY model /model

COPY crack_cracked_earth_2.jpg /crack_cracked_earth_2.jpg
COPY exported_plots/2015-01-13.png /exported_plots/2015-01-13.png
COPY test_df_robust_ohe.csv /test_df_robust_ohe.csv

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD streamlit run app.py --server.port $PORT
