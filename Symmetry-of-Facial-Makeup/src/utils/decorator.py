import time
import logging
from functools import wraps

def measuretime(func):
    """
    A decorator to measure the execution time of a function.

    Args:
        func (callable): The function to be wrapped.

    Returns:
        callable: The wrapped function with execution time measurement.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Get the logger for the wrapped function's module
        logger = logging.getLogger(func.__module__)

        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time

        # Convert elapsed time to hours, minutes, and seconds
        hours, remainder = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(remainder, 60)

        logger.info(f"Function '{func.__name__}' executed in {int(hours)}h {int(minutes)}m {seconds:.2f}s.")
        return result
    return wrapper