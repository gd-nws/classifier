FROM python:3-slim

RUN apt-get update && apt-get -y install cron libpq-dev gcc

RUN mkdir -p /app
WORKDIR /app

COPY ./start.sh /app/start.sh
COPY ./setup.sh /app/setup.sh

COPY ./classifier-cron /etc/cron.d/classifier-cron
RUN chmod 0644 /etc/cron.d/classifier-cron
RUN crontab /etc/cron.d/classifier-cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

# Install requirements
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt
RUN python3 -m nltk.downloader stopwords

# Copy project
COPY ./classifiers/ /app/classifiers/
COPY ./*.py /app/

# Run the command on container startup
CMD ./setup.sh