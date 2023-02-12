
from python:3.10


RUN mkdir rl_test
RUN mkdir mlruns
RUN mkdir logs

WORKDIR /rl_test/

ADD requirements.txt .
RUN pip install -r requirements.txt

COPY /custom_tetris ./custom_tetris
COPY /mlflow_cb ./mlflow_cb





