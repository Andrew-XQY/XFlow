"""Input/Output utilities for file operations."""

import shutil
from pathlib import Path
from typing import Union, Optional

def copy_file(source: Union[str, Path], target_dir: Union[str, Path], filename: Optional[str] = None) -> Path:
    """Copy a file to target directory.
    
    Args:
        source: Source file path
        target_dir: Target directory path  
        filename: Optional new filename (uses source filename if None)
        
    Returns:
        Path to the copied file
    """
    source_path = Path(source)
    target_dir_path = Path(target_dir)
    
    target_dir_path.mkdir(parents=True, exist_ok=True)
    target_filename = filename or source_path.name
    target_path = target_dir_path / target_filename
    
    shutil.copy2(source_path, target_path)
    return target_path

def create_directory(path: Union[str, Path]) -> Path:
    """Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
        
    Returns:
        Path to the created directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path