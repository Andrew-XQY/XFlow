#!/usr/bin/env python3
"""Build documentation with proper API structure"""

import os
import sys
from pathlib import Path


def main():
    """Build the documentation"""
    docs_dir = Path(__file__).parent
    os.chdir(docs_dir)

    print("Building XFlow documentation...")

    # Build docs
    result = os.system("sphinx-build -b html source build/html")

    if result == 0:
        # Create .nojekyll file to disable Jekyll on GitHub Pages
        nojekyll_path = docs_dir / "build" / "html" / ".nojekyll"
        nojekyll_path.touch()
        print("Created .nojekyll file for GitHub Pages")
        
        print("Documentation built successfully!")
        print(f"Open: {docs_dir}/build/html/index.html")
    else:
        print("Documentation build failed!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
