
from python:3.10


RUN mkdir rl_test
WORKDIR /rl_test/

ADD requirements.txt .
RUN pip install -r requirements.txt

RUN mkdir mlruns
RUN mkdir logs

COPY /rl_custom_games ./rl_custom_games






