import inspect
import os

def print_caller_directory():
    """
    Prints the directory path of the script that called this function.
    
    Useful for debugging or logging script origin in multi-file projects.
    """
    caller_frame = inspect.stack()[1]
    caller_file = os.path.abspath(caller_frame.filename)
    print("Caller script directory:", os.path.dirname(caller_file))