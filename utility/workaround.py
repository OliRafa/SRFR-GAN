import sys
import traceback
from functools import wraps
from multiprocessing import Process, Queue

def processify(func):
    '''Decorator to run a function as a process.
    Be sure that every argument and the return value
    is *pickable*.
    The created process is joined, so the code does not
    run in parallel.
    '''

    def process_func(q, *args, **kwargs):
        try:
            ret = func(*args, **kwargs)
        except Exception:
            ex_type, ex_value, tb = sys.exc_info()
            error = ex_type, ex_value, ''.join(traceback.format_tb(tb))
            ret = None
        else:
            error = None

        q.put((ret, error))

    # register original function with different name
    # in sys.modules so it is pickable
    process_func.__name__ = func.__name__ + 'processify_func'
    setattr(sys.modules[__name__], process_func.__name__, process_func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        q = Queue()
        p = Process(target=process_func, args=[q] + list(args), kwargs=kwargs)
        p.start()
        ret, error = q.get()
        p.join()

        if error:
            ex_type, ex_value, tb_str = error
            message = '%s (in subprocess)\n%s' % (ex_value.message, tb_str)
            raise ex_type(message)

        return ret
    return wrapper

import os
import cloudpickle
from multiprocessing import Pool

class RunAsCUDASubprocess:
    def __init__(self, num_gpus=0, memory_fraction=0.8):
        self._num_gpus = num_gpus
        self._memory_fraction = memory_fraction

    @staticmethod
    def _subprocess_code(num_gpus, memory_fraction, fn, args):
        # set the env vars inside the subprocess so that we don't alter the parent env
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see tensorflow issue #152
        try:
            import py3nvml
            num_grabbed = py3nvml.grab_gpus(num_gpus, gpu_fraction=memory_fraction)
        except:
            # either CUDA is not installed on the system or py3nvml is not installed (which probably means the env
            # does not have CUDA-enabled packages). Either way, block the visible devices to be sure.
            num_grabbed = 0
            os.environ['CUDA_VISIBLE_DEVICES'] = ""

        assert num_grabbed == num_gpus, 'Could not grab {} GPU devices with {}% memory available'.format(
            num_gpus,
            memory_fraction * 100)
        if os.environ['CUDA_VISIBLE_DEVICES'] == "":
            os.environ['CUDA_VISIBLE_DEVICES'] = "-1"  # see tensorflow issues: #16284, #2175

        # using cloudpickle because it is more flexible about what functions it will
        # pickle (lambda functions, notebook code, etc.)
        return cloudpickle.loads(fn)(*args)

    def __call__(self, f):
        def wrapped_f(*args):
            with Pool(1) as p:
                return p.apply(RunAsCUDASubprocess._subprocess_code, (self._num_gpus, self._memory_fraction, cloudpickle.dumps(f), args))

        return wrapped_f