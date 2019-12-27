#!/bin/bash

source /app/.env.sh

# the docker-compose variables should be available here
echo "DB_HOST = ${DB_HOST}"

/usr/local/bin/python3 /app/headline-analyser.py >> /var/log/cron.log 2>&1