"""
Timing function created by Paul McGuire and adapted to Python 3 / *nix
by Nicojo.
https://stackoverflow.com/questions/1557571/how-do-i-get-time-of-a-python-programs-execution/12344609#12344609

Just "import timing" and the script will print the start and end times, and the
overall elapsed time.

Call "timing.log" from within the program if there are significant stages within
the program you want to show.
"""
import atexit
import logging
from time import time, strftime, localtime
from datetime import timedelta

def secondsToStr(elapsed=None):
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))

def log(s, elapsed=None):
    line = "="*40
    logging.info(line)
    logging.info('{} - {}'.format(secondsToStr(), s))
    if elapsed:
        logging.info("Elapsed time: {}".format(elapsed))
    logging.info(line)

def endlog():
    end = time()
    elapsed = end-start
    log("End Program", secondsToStr(elapsed))

start = time()
atexit.register(endlog)
log("Start Program")

#import atexit
#from time import time, strftime, localtime
#from datetime import timedelta
#
#def secondsToStr(elapsed=None):
#    if elapsed is None:
#        return strftime("%Y-%m-%d %H:%M:%S", localtime())
#    else:
#        return str(timedelta(seconds=elapsed))
#
#def log(s, elapsed=None):
#    line = "="*40
#    print(line)
#    print(secondsToStr(), '-', s)
#    if elapsed:
#        print("Elapsed time:", elapsed)
#    print(line)
#    print()
#
#def endlog():
#    end = time()
#    elapsed = end-start
#    log("End Program", secondsToStr(elapsed))
#
#start = time()
#atexit.register(endlog)
#log("Start Program")