FROM python:3-slim

RUN apt-get update && apt-get -y install cron

RUN mkdir -p /app
WORKDIR /app

COPY ./start.sh /app/start.sh
RUN pip3 install pandas pymysql nltk vaderSentiment requests
RUN python3 -m nltk.downloader stopwords

COPY ./classifiers/ /app/classifiers/
COPY ./headline-analyser.py /app/headline-analyser.py

# Copy hello-cron file to the cron.d directory
COPY ./classifier-cron /etc/cron.d/classifier-cron

# Give execution rights on the cron job
RUN chmod 0644 /etc/cron.d/classifier-cron

# Apply cron job
RUN crontab /etc/cron.d/classifier-cron

# Create the log file to be able to run tail
RUN touch /var/log/cron.log

COPY ./setup.sh /app/setup.sh

# Run the command on container startup
CMD ./setup.sh