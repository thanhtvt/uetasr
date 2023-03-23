FROM tensorflow/tensorflow:2.10.1-gpu

COPY . /uetasr
WORKDIR /uetasr

RUN apt-get update && apt-get install -y git cmake

RUN pip install -e  .