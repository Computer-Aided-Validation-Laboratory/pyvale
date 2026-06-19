import time
from pathlib import Path
from pyvale.mooseherder import (MooseConfig,
                                MooseRunner)


MOOSE_FILE = "platehole3d.i"
MOOSE_PATH = Path.cwd() / MOOSE_FILE

USER_DIR = Path.home()


def main() -> None:
    config = {"main_path": USER_DIR / "moose",
              "app_path": USER_DIR / "proteus",
              "app_name": "proteus-opt"}

    moose_config = MooseConfig(config)
    moose_runner = MooseRunner(moose_config)

    moose_runner.set_run_opts(n_tasks = 4,
                              n_threads = 1,
                              redirect_out = False)

    moose_start_time = time.perf_counter()
    moose_runner.run(MOOSE_PATH)
    moose_run_time = time.perf_counter() - moose_start_time

    print()
    print("="*80)
    print(f"MOOSE run time = {moose_run_time:.3f} seconds")
    print("="*80)
    print()

if __name__ == "__main__":
    main()

