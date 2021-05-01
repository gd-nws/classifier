FROM python:3-slim as builder

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && apt-get clean

RUN mkdir -p /app
WORKDIR /app

# Install requirements
COPY ./requirements.txt /app/requirements.txt
RUN pip install --user -r requirements.txt
RUN python3 -m nltk.downloader stopwords

# Copy project
COPY ./classifiers/ /app/classifiers/
COPY ./*.py /app/

WORKDIR /app
ENV PATH=/root/.local/bin:$PATH

# Run the command on container startup
CMD /usr/local/bin/python3 /app/headline-analyser.py --persist