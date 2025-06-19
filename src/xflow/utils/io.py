"""Input/Output utilities for file operations."""

import shutil
from pathlib import Path
from typing import Optional
from .typing import PathLikeStr

def copy_file(source_path: PathLikeStr, target_path: PathLikeStr, filename: Optional[str] = None) -> Path:
    """Copy a file to target directory.
    
    Args:
        source_path: Source file path
        target_path: Target directory path  
        filename: Optional new filename (uses source filename if None)
        
    Returns:
        Path to the copied file
    """
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    target_path.mkdir(parents=True, exist_ok=True)
    target_filename = filename or source_path.name
    target_path = target_path / target_filename
    
    shutil.copy2(source_path, target_path)
    return target_path

def create_directory(path: PathLikeStr) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path to the created directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path