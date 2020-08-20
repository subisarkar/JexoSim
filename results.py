"""
Created on Fri Aug  7 08:42:22 2020
"""

from jexosim.run_files import results
import sys
import numpy as np

def go():
    input_file = sys.argv[1]
    results.run(input_file)

if __name__ == '__main__':
    go()
