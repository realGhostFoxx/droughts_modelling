FROM python:3.8.6-buster

COPY droughts_modelling /droughts_modelling
COPY requirements.txt /requirements.txt
COPY data /data 
COPY app.py /app.py 
COPY model /model 
COPY crack_cracked_earth_2.jpg /crack_cracked_earth_2.jpg
RUN pip install -r requirements.txt

CMD streamlit run app.py  --server.port 8080