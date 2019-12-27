#!/bin/bash

printenv | sed 's/^\(.*\)$/export \1/g' >> ./.env.sh
chmod +x ./.env.sh

cron && tail -f /var/log/cron.log