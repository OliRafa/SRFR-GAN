"""Helper module to deal with errors in functional programming pipelines.

Works encapsulating any result of a function inside the Result class, and
decorating the next function to be called with the bind decorator."""
from functools import namedtuple, wraps

class Result():
    """Base class for result output for any function within a pipeline.

    In combination with 'bind' Decorator, does the implementation of "Railway
    Oriented Programming" for error handling in function pipelines.

    ### Parameters:
        result - Result of the computation within the function, either
            'Success' or 'Failure'.
        payload: Return of the function.
    """
    __slots__ = ('_result', '_payload', '_args')

    def __init__(self, result, payload, args=None):
        self._result = result
        if args:
            self._payload = payload
        else:
            self._payload = (payload, )
        self._args = args

    def get_result(self):
        """Getter for Result.

        ### Returns: the result.
        """
        return self._result

    def get_payload(self):
        """Getter for Payload.

        ### Returns: the payload.
        """
        if self._args:
            return self._payload, self._args
        return self._payload

def bind(function):
    """Decorator for function error handling within a pipeline.

    Works with the Result class, making a decision of calling the decorated
    function in case of result = 'Success', or passing-through in case of
    result = 'Failure'.
    """
    @wraps(function)
    def _bind(Result, *args, **kwargs):
        """Inner class to handle the switch case of result."""
        if Result.get_result() == 'Success':
            #print(*Result.get_payload())
            return function(*Result.get_payload(), *args, **kwargs)
        return Result
    return _bind

ImageContainer = namedtuple('ImageContainer', ('image', 'image_path'))