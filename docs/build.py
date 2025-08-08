#!/usr/bin/env python3
"""Build documentation with proper API structure"""

import sys
import subprocess
from pathlib import Path

def main() -> int:
    docs_dir = Path(__file__).parent.resolve()
    source = docs_dir / "source"
    outdir = docs_dir / "build" / "html"
    outdir.mkdir(parents=True, exist_ok=True)

    if not (source / "index.rst").exists() and not (source / "index.md").exists():
        print(f"[error] missing index.(rst|md) in {source}")
        return 2

    print("Building XFlow documentation...")
    # -W: treat warnings as errors (optional), --keep-going: continue building
    cmd = [sys.executable, "-m", "sphinx", "-b", "html", str(source), str(outdir)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print("[error] Documentation build failed!")
        return result.returncode

    # Disable Jekyll for GitHub Pages
    (outdir / ".nojekyll").touch()
    print("Created .nojekyll file for GitHub Pages (source from Action do not need this)")
    
    # Create a redirect file for the root if needed (helps with GitHub Pages)
    if not (outdir / "index.html").exists():
        print("[warning] No index.html generated - this might cause GitHub Pages issues")
    
    print("Documentation built successfully!")
    print(f"Open: {outdir / 'index.html'}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
