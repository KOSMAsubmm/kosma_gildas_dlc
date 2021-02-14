from __future__ import print_function
import pyclass
import pgutils
from kosma_py_lib.pandas_index import get_index
from sicparse import OptionParser
import pandas as pd
import copy
import numpy as np
import scipy.stats as stats
import math
from kosma_py_lib.noise_statistic_utilities import SpectraStatistic

def main():
    parser = OptionParser()
    parser.add_option("-a", "--alpha", dest="alpha", nargs=1, default="2")
    parser.add_option("-c", "--avg_channels", dest="avg_channels", nargs=1, default="40")

    try:
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "NOISE_STATS", "Invalid option")
        pyclass.sicerror()
        return
    if (not pyclass.gotgdict()):
        pyclass.get(verbose=False)
    pyclass.comm("set var user")
    pyclass.comm("import sofia")

    spectrum = SpectraStatistic()
    spectrum.iterative_smooth()

main()
