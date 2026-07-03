# ================================================================================
# pyvale: the python validation engine
# License: MIT
# Copyright (C) 2025 The Computer Aided Validation Team
# ================================================================================

import os
import sys
from datetime import datetime

def print_pyvale_banner():
    print("                                 #              ##            ####                       ")
    print("                                 #+#          #-+#            #--+                       ")
    print("                                 ##-##      ##--#             #--+                       ")
    print("                                  #.-+#    #++.+#   ### ####  #--+     ####              ")
    print("                                  #+---#  #.---# ##-----+--#  #--+  ##------#            ")
    print("                                   #.---##.-.-+# +--+###+--#  #--+ #+--####--#           ")
    print("                                   #++--+-...+#  +--## #+--#  #--+ #+--#######           ")
    print("                                    #+-+#--.-#   #+--------#  #--+  #----+---#           ")
    print("                                    #...+..+#      ##+######  ####    ##++##             ")
    print("                                  #+++---+-#                                             ")
    print("     #####+++##  #####  #####   ##-.----.--++--.------------+-----.-##                   ")
    print("     #---------#  +--+##+--#   #+-.--.-.--..-.-..+.--.+.-..--#..-.--.-+.-##              ")
    print("     #---#  #---# #+--++--##   #--..-.---.-..-------.-----.#+..----.-.#-+-#              ")
    print("     #---###+--+    +----#      #+---#-+-.-..-.---.-++-+..-+....--.-.-..#-+##            ")
    print("     #--------+#    #+--#       ##+...-.----..-----------..-++---.--.----+-.--+#         ")
    print("     #---####      #+--#        ###+---.-..-+.--.-.-+-..-----------++#..+.++##..-##      ")
    print("     #---#        #+--#              ##########+++#+#++###########   #----#   #+++#      ")
    print("     #####       #####                                                #+.-#     ##       ")
    print("                                                                        ##               ")

def info(msg: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    print(f"[{timestamp}] {msg}")


def info_out(label: object, value: object) -> None:
    total_width = 80
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    lhs = f"[{timestamp}] {label}"
    rhs = str(value)
    padding = 1

    if len(lhs) + len(rhs) < total_width:
        padding = total_width - len(lhs) - len(rhs)

    print(lhs + " " * padding + rhs)


def check_output_directory(output_basepath: str,
                           output_prefix: str, debug_level: int) -> None:
    """
    Check for existing output files in a directory and prompt user confirmation before overwriting.

    This function verifies whether the specified output directory exists and checks for any existing
    files that match a given prefix and have `.csv` or `.dic2d` extensions. If such files are found,
    a list is displayed and the user is prompted to confirm whether to continue. If the user declines,
    the program exits to prevent data loss.

    Parameters
    ----------
    output_basepath : str
        Path to the output directory where files are or will be saved.
    output_prefix : str
        Filename prefix used to identify potential conflicting output files.
    debug_level: int
        Determines how much information to provide in console output.

    Raises
    ------
    SystemExit
        If the output directory does not exist or the user chooses not to proceed after
        being warned about existing files.
    """
    
    # check if there's output files
    try:
        files = os.listdir(output_basepath)
    except FileNotFoundError:
        print("")
        print(f"Output directory '{output_basepath}' does not exist.")
        sys.exit(1)

    # Check for any matching files
    conflicting_files = [
        f for f in files 
        if f.startswith(output_prefix) and (f.endswith(".csv") or f.endswith(".dic2d"))]

    if conflicting_files:
        conflicting_files.sort()
        if (debug_level>0):
            info("WARNING: files with matching prefix in output dir")


        ###### TURNING USER INPUT OFF FOR NOW ######
        # user_input = input("Do you want to continue? (y/n): ").strip().lower()

        # if user_input not in ("y", "yes", "Y", "YES"):
        #     print("Aborting to avoid overwriting data in output directory.")
        #     exit(0)
