FROM python:3.8.6-buster

COPY api /api
COPY api/fast.py /api/fast.py
COPY api/__init__.py /api/__init__.py
COPY droughts_modelling /droughts_modelling
COPY requirements.txt /requirements.txt
COPY Makefile /Makefile
COPY setup.py /setup.py
COPY api /api
COPY raw_data /raw_data 

RUN pip install -r requirements.txt

CMD uvicorn api.fast:app --host 0.0.0.0 --port $PORT