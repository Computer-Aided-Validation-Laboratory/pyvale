"""
==============================================================================
EXAMPLE: Create moose-config.json

Author: Lloyd Fletcher
==============================================================================
"""
from pathlib import Path
from mooseherder import MooseConfig


def main() -> None:
    """main: create moose config json
    """
    config = {'main_path': Path.home()/ 'moose',
            'app_path': Path.home() / 'proteus',
            'app_name': 'proteus-opt'}

    moose_config = MooseConfig(config)

    save_path = Path.cwd() / 'moose-config.json'
    moose_config.save_config(save_path)


if __name__ == "__main__":
    main()

