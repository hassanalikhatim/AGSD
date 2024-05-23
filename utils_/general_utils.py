import os
import numpy as np


def confirm_directory(path):
    if not os.path.isdir(path):
        os.makedirs(path)
    return


def get_memory_usage():
    # Importing the library to measure RAM usage
    import psutil
    return psutil.virtual_memory()[2]

