"""This module provides functions to easy timing, made for logging.
This was forked from Nicojo's function (which is an adaptation to Python 3 / \
*nix from Paul McGuire's function).
Original function: \
https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution/12344609#12344609
"""
import atexit
import logging
from datetime import timedelta
from time import time, strftime, localtime

LOGGER = logging.getLogger(__name__)

class TimingLogger():
    """Timing Logger base class.

    To start the timing logger, just call TimingLogger().start(), and when the\
 program ends it will automatically log the end and elapsed time.\\
    To start a timing logger for other than the whole program, call\
 TimingLogger().start(function_name) passing the function's name, and, to end,\
 call TimingLogger().end(function_name).

    ### Methods
        start: Starts time logging.
        end: Ends time logging.
    """
    def __init__(self):
        self._start = {}
        self._mean_times = {}
        atexit.register(self.end)

    def start(self, function_name: str = None) -> None:
        """Starts time logging.
        If the name of the function is provided, starts logging time for the\
 given function, otherwise it will start logging for the whole program.

        ### Parameters
            function_name: Name of the function to log the time.
        """
        if function_name:
            self._start[function_name] = time()
            self._log(f'Start {function_name}')
        else:
            self._start['program'] = time()
            self._log(f'Start Main Program')

    def _secondsToStr(self, elapsed: float = None) -> str:
        if elapsed is None:
            return strftime('%Y-%m-%d %H:%M:%S', localtime())
        else:
            return str(timedelta(seconds=elapsed))

    def _log(self, s, elapsed: float = None) -> None:
        line = '-' * 40
        LOGGER.info(f' {line}')
        LOGGER.info(f' {self._secondsToStr()} - {s}')
        if elapsed:
            LOGGER.info(f' Elapsed time: {elapsed}')
        LOGGER.info(f' {line}')

    def _add_to_mean(self, function_name: str, timing: float) -> None:
        if not self._mean_times.get(function_name):
            self._mean_times[function_name] = {
                'times': [],
                'count': 0,
            }
        print(function_name)
        self._mean_times[function_name]['times'].append(timing)
        self._mean_times[function_name]['count'] += 1

    def calculate_mean(self, function_name: str) -> float:
        """Calculates and logs the mean time for a giving function.

        Before calculate the mean, make shure to call end() function, passing\
 function_name and mean=True, otherwise mean data won't be recorded.

        ### Parameters
            function_name: Function name to have the mean time calculated.

        ### Returns
            Value for the mean time.
        """
        mean = sum(self._mean_times[function_name]['times']) / \
            self._mean_times[function_name]['count']
        string_mean = self._secondsToStr(mean)
        LOGGER.info(f' Mean time for {function_name}: {string_mean}')
        return mean

    def end(
            self,
            function_name: str = None,
            mean: bool = False,
        ) -> float:
        """Ends time logging.
        If the name of the function is provided, ends logging time for the\
 given function, otherwise it will end logging for the whole program.

        ### Parameters
            function_name: Name of the function to log the time.
            mean: If the elapsed time will be registered for latter calculation\
 of the mean time or not.
            return_value: If the elapsed value will be returned or not.

        ### Returns
            The elapsed time
        """
        if not function_name and not self._start.get('program'):
            return

        ending_time = time()
        if function_name:
            elapsed = ending_time - self._start[function_name]
            self._log(f'End {function_name}', self._secondsToStr(elapsed))
        else:
            elapsed = ending_time - self._start['program']
            self._log('End Main Program', self._secondsToStr(elapsed))

        if mean:
            self._add_to_mean(function_name, elapsed)

        return elapsed
