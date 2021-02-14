
from __future__ import print_function
import copy
import pickle
import subprocess
import os

import pyclass
from sicparse import OptionParser



def main():

    # Input parameters
    parser = OptionParser()
    parser.add_option("-o", "--output", dest="output", nargs=1, default=False)
    try:
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "PCA", "Invalid option")
        pyclass.sicerror()
        return
    if (not pyclass.gotgdict()):
        pyclass.get(verbose=False)
    print(options.output)
    if os.path.isdir(options.output):
        print("path exists --{}-- ".format(options.output))
    if not os.path.isdir(options.output):
        print("path does exist--{}-- ".format(options.output))

main()
