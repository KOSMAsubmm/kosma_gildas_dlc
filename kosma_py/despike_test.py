import glob
import math
from collections import deque
from copy import deepcopy
from random import random

import numpy as np
import pandas
import scipy
from scipy import signal

import pyclass
from sicparse import OptionParser


def moving_average(interval, window_size):
    window = np.ones(int(window_size))/float(window_size)
    return np.convolve(interval, window, 'same')

def main():
    parser = OptionParser()
    parser.add_option("-l", "--length", dest="lenght", nargs=1, default=60)
    parser.add_option("-c", "--cutoff", dest="cutoff", nargs=1, default=5)
    parser.add_option("-n", "--noise", dest="noise", nargs=1, default=None)
    try:
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "DESPIKE", "Invalid option")
        pyclass.sicerror()
        return
    if (not pyclass.gotgdict()):
        pyclass.get(verbose=False)

    data = deepcopy(pyclass.gdict.ry.__sicdata__)

    wavelet_length = int(options.lenght)
    cutoff = int(options.cutoff)
    noise = options.noise

    cleaned_data = deepcopy(data)
    series = pandas.Series(data)
    tmp =deepcopy(series)
    mov_median = tmp.rolling(window=wavelet_length, center=False).median()
    flat_series =  series.sub(mov_median)
    moving_std = flat_series.rolling(window=wavelet_length, center=False).std()
    print moving_std[10000]
    positive_outliers = np.where((flat_series < -1 * cutoff * moving_std))
    negative_outliers = np.where((flat_series > cutoff * moving_std))

    outliers = np.concatenate([positive_outliers, negative_outliers], 1)

    print outliers

    cleaned_data[outliers] = np.nan
    pyclass.gdict.ry =  cleaned_data
    return


if __name__ == "__main__":
    main()
