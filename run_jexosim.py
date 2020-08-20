from jexosim.run_files import run_jexosim
import sys
import numpy as np

def go():
    input_file = sys.argv[1]
    run_jexosim.run(input_file)

if __name__ == '__main__':
    go()