"""
================================================================================
License: MIT
Copyright (C) 2024 The Computer Aided Validation Team
================================================================================
"""
import time
from pathlib import Path
from pyvale.mooseherder import GmshRunner

GMSH_FILE = "platewithhole3d.geo"
GMSH_PATH = Path.cwd() / GMSH_FILE

PARSE_ONLY = False

USER_DIR = Path.home()

def main() -> None:
    gmsh_runner = GmshRunner(USER_DIR / "gmsh/bin/gmsh")
    print(GMSH_PATH)
    gmsh_start = time.perf_counter()
    gmsh_runner.run(GMSH_PATH,parse_only=PARSE_ONLY)
    gmsh_run_time = time.perf_counter()-gmsh_start

    print()
    print("="*80)
    print(f"Gmsh run time = {gmsh_run_time:.2f} seconds")
    print("="*80)
    print()

if __name__ == "__main__":
    main()

