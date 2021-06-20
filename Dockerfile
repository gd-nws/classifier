FROM python:3.8-slim

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && apt-get clean

RUN mkdir -p /app
WORKDIR /app

RUN pip3 install cython 
RUN pip3 install numpy
RUN pip3 install scipy

# Install requirements
COPY ./requirements.txt /app/requirements.txt
RUN pip3 install -r requirements.txt
RUN python3 -m nltk.downloader stopwords

# Copy project
COPY ./classifiers/ /app/classifiers/
COPY ./*.py /app/

WORKDIR /app

# Run the command on container startup
CMD /usr/local/bin/python3 /app/headline-analyser.py --persist