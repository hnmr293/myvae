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
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        #logging.info(f'start watchdog {prog} [PID={self.process.pid}]')
        print(f'start watchdog {prog} [PID={self.process.pid}]')
    
    def close(self):
        try:
            self.process.send_signal(signal.SIGINT)
        except Exception as e:
            logging.error(e)
            #print(e, file=sys.stderr)
    
    @property
    def pid(self):
        return self.process.pid
    
    def notify1(self):
        os.kill(self.process.pid, signal.SIGUSR1)
    
    def notify2(self):
        os.kill(self.process.pid, signal.SIGUSR2)
