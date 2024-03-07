import rq
import signal
import os
from redis import Redis
from resdis_connection import host

for w in rq.Worker.all(connection=Redis(host)):
    os.kill(w.pid, signal.SIGKILL)