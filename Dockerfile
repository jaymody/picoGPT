FROM python:3.10.10-slim-bullseye
WORKDIR /picogpt
ADD . /picogpt
RUN apt-get update && apt-get -y install gcc
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get -y autoremove gcc
ENTRYPOINT ["python", "gpt2.py"]
