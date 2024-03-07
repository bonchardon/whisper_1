#!/bin/sh

python -m kill_workers && (rq worker --with-scheduler -u "redis://10.100.70.208:6379" & nohup python -m main) > log.txt&