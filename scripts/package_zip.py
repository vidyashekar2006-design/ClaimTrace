"""
ClaimTrace: Project Packaging Script
Packages the full Python mini-project into ClaimTrace.zip
"""

import os
import zipfile
import sys

def package_project():
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    zip_destinations = [
        os.path.join(root_dir, "ClaimTrace.zip"),
        os.path.join(root_dir, "public", "ClaimTrace.zip"),
    ]

    included_dirs = ["src", "data", "artifacts", "notebooks", "scripts", "tests"]
    included_files = ["app.py", "requirements.txt", "README.md", ".gitignore"]

    # Verify key files exist before packaging
    for f in included_files:
        p = os.path.join(root_dir, f)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing required file: {p}")

    for dest in zip_destinations:
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with zipfile.ZipFile(dest, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Add top-level files
            for fname in included_files:
                fpath = os.path.join(root_dir, fname)
                arcname = os.path.join("ClaimTrace", fname)
                zipf.write(fpath, arcname)

            # Add directories
            for dirname in included_dirs:
                dirpath = os.path.join(root_dir, dirname)
                if not os.path.exists(dirpath):
                    continue
                for root, _, files in os.walk(dirpath):
                    for file in files:
                        if file.endswith((".pyc", ".DS_Store")):
                            continue
                        if "__pycache__" in root:
                            continue
                        # Do not include typescript/react files from src in python zip
                        if dirname == "src" and file.endswith((".tsx", ".ts", ".css")):
                            continue
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, root_dir)
                        arcname = os.path.join("ClaimTrace", rel_path)
                        zipf.write(full_path, arcname)

        size_kb = os.path.getsize(dest) / 1024
        print(f"Packaged {dest} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    package_project()
