import time
from collections import defaultdict


class Timer():

    def __init__(self, desc: str):
        self._desc = desc
    def __enter__(self):
        self._enter_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Time for section {self._desc} is {time.time()-self._enter_time}")

class MultiTimer():

    def __init__(self, desc: str):
        self._desc = desc
    def __enter__(self):
        self._enter_time = time.time()
        self._sums = defaultdict(lambda:0)
        self._last_starts = {}
        return self

    def start(self, key:str="default"):
        self._last_starts[key] = time.time()

    def stop(self, key:str="default"):
        self._sums[key] += time.time() - self._last_starts[key]
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Time for section {self._desc} is {dict(self._sums)} over the course of {time.time()-self._enter_time}s")