"""
Script to generate __init__.py files from API registry. Simply run: python path_to_dir/generate_api.py
"""

import sys
from pathlib import Path


def clean_init_files():
    """Clean all __init__.py files to avoid circular imports during generation"""
    src_dir = Path(__file__).parent.parent / "src"

    # Clean main package __init__.py
    main_init = src_dir / "xflow" / "__init__.py"
    if main_init.exists():
        print(f"Cleaning {main_init}")
        with open(main_init, "w") as f:
            f.write(
                '"""Auto-generated API exports"""\n# Temporarily empty during generation\n'
            )

    # Clean known subpackage __init__.py files
    known_packages = ["data", "models", "utils", "trainers"]  # Add your known packages
    for package_name in known_packages:
        package_init = src_dir / "xflow" / package_name / "__init__.py"
        if package_init.exists():
            print(f"Cleaning {package_init}")
            with open(package_init, "w") as f:
                f.write(
                    '"""Auto-generated API exports"""\n# Temporarily empty during generation\n'
                )


def generate_all_apis():
    """Generate all __init__.py files"""
    # NOW it's safe to import from xflow
    from xflow._api_registry import CORE_API, PACKAGE_API, generate_init

    src_dir = Path(__file__).parent.parent / "src"

    # Generate main package __init__.py
    main_init = src_dir / "xflow" / "__init__.py"
    main_content = generate_init(CORE_API, "xflow", include_version=True)  # <-- add flag
    print(f"Generating {main_init}")
    with open(main_init, "w") as f:
        f.write(main_content)

    # Generate subpackage __init__.py files (no version)
    for package_name, api_dict in PACKAGE_API.items():
        package_init = src_dir / "xflow" / package_name / "__init__.py"
        package_init.parent.mkdir(parents=True, exist_ok=True)
        package_content = generate_init(api_dict, f"xflow.{package_name}", include_version=False)
        print(f"Generating {package_init}")
        with open(package_init, "w") as f:
            f.write(package_content)


if __name__ == "__main__":
    # Add src to path
    src_path = Path(__file__).parent.parent / "src"
    sys.path.insert(0, str(src_path))

    print("Step 1: Cleaning existing __init__.py files...")
    clean_init_files()

    print("\nStep 2: Generating new __init__.py files...")
    generate_all_apis()

    print("\nAPI generation complete!")
