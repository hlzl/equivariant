import sys
import numpy as np
from pickle import load


def closest_divisor(number, div):
    """
    Get next closest divisor to given number and divisor. Take second closest
    divisor if initial one results in number smaller than 3.

    We set the value 3 heuristically as group normalization with groups of less
    than 4 channels seem to perform worse.
    """
    all_divs = [i for i in range(1, number + 1) if number % i == 0]
    if len(all_divs) < 2:
        closest_div = all_divs[0]
    else:
        idx = np.argsort(np.abs(np.array(all_divs) - div))[:2]
        # Check next closest divisor if result is too small
        if int(number / all_divs[idx[0]]) > 3:
            return all_divs[idx[0]]
        closest_div = all_divs[idx[1]]
    # Set divisor to 1 if result is too small
    if int(number / closest_div) <= 3:
        return 1
    return closest_div


def unpickle(file):
    with open(file, "rb") as open_file:
        data_dict = load(open_file)
    return data_dict


class PrintAndSaveFile:
    def __init__(self, file_path):
        self.file = open(file_path, "w")
        self.stdout = sys.stdout

    def write(self, text):
        self.file.write(text)
        self.stdout.write(text)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        self.file.close()
