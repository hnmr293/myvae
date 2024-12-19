import sys
import os
import signal
import subprocess
from typing import Any
import logging


class Watchdog:
    def __init__(self, prog: str, args: list[Any]):
        python = sys.executable
        args = [str(x) for x in args]
        self.process = subprocess.Popen(
            [python, prog, *args],
            start_new_session=True,
        )
        print(f'start watchdog {prog} [PID={self.process.pid}]')
    
    def close(self):
        try:
            self.process.send_signal(signal.SIGINT)
        except Exception as e:
            logging.error(e)
    
    @property
    def pid(self):
        return self.process.pid
    
    def notify1(self):
        try:
            os.kill(self.process.pid, signal.SIGUSR1)
        except Exception as e:
            logging.error(e)
    
    def notify2(self):
        try:
            os.kill(self.process.pid, signal.SIGUSR2)
        except Exception as e:
            logging.error(e)
