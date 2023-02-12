
from python:3.10


RUN mkdir rl_test
ADD requirements.txt /rl_test/

ADD custom_tetris /rl_test/
ADD mlflow_cb /rl_test/

WORKDIR /rl_test/

RUN pip install -r requirements.txt

