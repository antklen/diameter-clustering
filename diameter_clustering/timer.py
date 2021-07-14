"""
Timer which saves history of runs.
"""

import logging
import time
from contextlib import contextmanager


@contextmanager
def timer(name, disable=False):
    """Simple timer as context manager."""

    start = time.time()
    yield
    if not disable:
        logging.info(f'[{name}] done in {(time.time() - start)*1000:.1f} ms')


class TimerWithHistory:
    """Timer as context mamager which saves history.

    This timer should be initialized and then used as context manager.
    After each run it appends execution time to list with history.
    Different runs could have different names and history is saved as dict
    with separate key for each name.

    Args:
        default_name (str): Default name for given run.
        disable (bool): If True then disable timer.

    Example:
        timer = TimerWithHistory()
        with timer():
            time.sleep(1)
        with timer(name='first'):
            time.sleep(2)
        # get history
        hist1, hist2 = timer.history['default'], timer.history['first']
    """

    def __init__(self, default_name='default', disable=False):
        self._start = None
        self.history = {}
        self.name = default_name
        self.default_name = default_name
        self.disable = disable

    def start(self):
        """Start timer."""

        if self._start is not None:
            raise RuntimeError('Timer already started...')
        self._start = time.perf_counter()

    def stop(self):
        """Stop timer and save result to history."""

        if self._start is None:
            raise RuntimeError('Timer not yet started...')
        elapsed = time.perf_counter() - self._start
        if self.history.get(self.name):
            self.history[self.name].append(elapsed)
        else:
            self.history[self.name] = [elapsed]
        self._start = None

    def __enter__(self):
        if not self.disable:
            self.start()
        return self

    def __exit__(self, *args):
        if not self.disable:
            self.stop()

    def __call__(self, name=None):
        if not self.disable:
            self.name = name or self.default_name
        return self
