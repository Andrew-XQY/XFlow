import inspect
import os
import sys
import __main__
from pathlib import Path

def print_caller_directory():
    """
    Prints the directory path of the script that called this function.
    
    Useful for debugging or logging script origin in multi-file projects.
    """
    caller_frame = inspect.stack()[1]
    caller_file = os.path.abspath(caller_frame.filename)
    print("Caller script directory:", os.path.dirname(caller_file))
    

def get_base_dir() -> Path:
    """
    Returns the directory path of the calling context with enhanced robustness.
    """
    # 1. Direct script execution: __main__.__file__ exists
    try:
        main_file = getattr(__main__, '__file__', None)
        if main_file and os.path.exists(main_file):
            return Path(main_file).parent.resolve()
    except Exception:
        pass

    # 2. Check if running as frozen executable
    try:
        if getattr(sys, 'frozen', False):
            # Fix: Both branches do the same thing, could be simplified
            return Path(sys.executable).parent.resolve()
    except Exception:
        pass

    # 3. Fallback: inspect stack for first external caller
    try:
        # Fix: Remove unused variable
        current_file = Path(__file__).resolve()
        
        for frame_info in inspect.stack()[1:]:  # skip current frame
            filename = frame_info.filename
            # Skip interactive frames, this module, and built-ins
            if (filename.startswith('<') or 
                filename.startswith('[') or  # some REPLs use brackets
                Path(filename).resolve() == current_file):
                continue
            
            file_path = Path(filename)
            if file_path.exists():
                return file_path.parent.resolve()
                
    except Exception:
        pass

    # 4. Try sys.argv[0] if available
    try:
        if sys.argv and sys.argv[0]:
            script_path = Path(sys.argv[0])
            if script_path.exists() and script_path.is_file():
                return script_path.parent.resolve()
    except Exception:
        pass

    # 5. Ultimate fallback: current working directory
    return Path(os.getcwd()).resolve()