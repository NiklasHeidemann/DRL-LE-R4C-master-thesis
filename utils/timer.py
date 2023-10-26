import time
from collections import defaultdict

"""
Usage: 
with Timer("some section"):
    # do stuff

The time needed for everything in the with block will be printed after the block is done with desc as identifier
"""
class Timer():

    def __init__(self, desc: str):
        self._desc = desc
    def __enter__(self):
        self._enter_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Time for section {self._desc} is {time.time()-self._enter_time}")

"""
Usage:
with MultiTimer("some section") as timer:
    # do stuff
    timer.start("some key")
    # do some-key-stuff
    timer.start("time stopping of timer")
    timer.stop("some key")
    timer.stop("time stopping of timer")
    # do more stuff and repeat for other keys

# the times between the start and stop for each key will be printed after the whole with-block is done with desc as identifier.
# the time for the whole block will be printed as well
# A key can be used multiple times, the times will be summed up
"""
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