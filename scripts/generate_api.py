"""
Script to generate __init__.py files from API registry.
"""

import os
import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from xflow._api_registry import CORE_API, PACKAGE_API, generate_init_content

def generate_all_apis():
    """Generate all __init__.py files"""
    src_dir = Path(__file__).parent.parent / "src"
    
    # Generate main package __init__.py
    main_init = src_dir / "xflow" / "__init__.py"
    main_content = generate_init_content(CORE_API, "xflow")
    
    print(f"Generating {main_init}")
    with open(main_init, "w") as f:
        f.write(main_content)
    
    # Generate subpackage __init__.py files
    for package_name, api_dict in PACKAGE_API.items():
        package_init = src_dir / "xflow" / package_name / "__init__.py"
        package_init.parent.mkdir(parents=True, exist_ok=True)
        
        package_content = generate_init_content(api_dict, f"xflow.{package_name}")
        
        print(f"Generating {package_init}")
        with open(package_init, "w") as f:
            f.write(package_content)

if __name__ == "__main__":
    generate_all_apis()
    print("API generation complete!")