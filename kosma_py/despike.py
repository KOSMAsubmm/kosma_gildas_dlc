import glob
import math
from collections import deque
from copy import deepcopy
from random import random
import os
import numpy as np
#import scipy
#from matplotlib import pyplot as plt
#from scipy import signal
import pandas
import pyclass
from sicparse import OptionParser
import re


def determine_blanks_and_fast_changes(data, wavelet_length=2, cutoff=8):
    cleaned_data = data.copy()
    kernel = np.asarray([1., 1., -1. - 1.])
    convolved_data = abs(np.convolve(data, kernel, "same"))
    # After this convolution all channels that have the same values as their
    # surrounding four value are zero. I assume that this is only true for
    # blanked channels or channels in saturation. Thus we now can get also an
    # idea of the blank value to use

    cleaned_data[np.where(convolved_data == 0.0)] = np.nan
    # print "Blanking value in", np.unique([int(point) for point in data[np.where(convolved_data == 0.0)]])
    convolved_data[:2 * wavelet_length] = np.nan
    convolved_data[-2 * wavelet_length:] = np.nan
    last_std = 0
    convolved_std = np.nanstd(convolved_data)
    while last_std != convolved_std:
        outliers = np.where(convolved_data > cutoff * convolved_std)
        for idx in outliers[0]:
            cleaned_data[idx - wavelet_length *
                         2:idx + wavelet_length * 2] = np.nan
            convolved_data[idx - wavelet_length *
                           2:idx + wavelet_length * 2] = np.nan
        last_std = convolved_std
        convolved_std = np.nanstd(convolved_data)
    return cleaned_data, convolved_data


def moving_average(interval, window_size):
    window = np.ones(int(window_size)) / float(window_size)
    return np.convolve(interval, window, 'same')


def main():
    description = """
default flag found channels as bad
  $ despike
default flag found channels write channels to associated_array
  $ despike /use_associated_array
use a lookup table and write channels to associated array
  $ despike /lookup_table data/despike_lookup_table.pkl /use_associated_array
blank channels from command line and write to an associated array
  $ despike /blank_ranges 7000 8000 /blank_ranges 1000 3000  /use_associated_array
blank channels from command line and overwrite with noise
  $ despike /blank_ranges 7000 8000 /blank_ranges 1000 3000  /noise
"""
    parser = OptionParser(description)
    parser.add_option("-l", "--length", dest="length",
                      nargs=1,
                      default=40,
                      help="channel length of wavelet to convolve")
    parser.add_option("-c", "--cutoff", dest="cutoff",
                      nargs=1,
                      default=5,
                      help="threshold for standard deviation around"
                      "rolling standard deviation")
    parser.add_option("-n", "--noise", dest="noise",
                      action="store_true", default=False,
                      help="replace detected spurs with local noise")
    parser.add_option("-a", "--use_associated_array",
                      dest="use_associated_array",
                      action="store_true",
                      default=False,
                      help="use associated array to stored blanking postions")
    parser.add_option("-t", "--lookup_table", dest="lookup_table",
                      default=None,
                      help="use lookup table to apply spurs")
    parser.add_option("-s", "--smooth", dest="smooth_data_before", default=0,
                      type="int",
                      help=("run smooth routine on smoothed version of data, "
                            "box width is the option"))
    parser.add_option("-p", "--plot", dest="plot",
                      default=False,
                      action="store_true",
                      help=("show plot of detected spurs, "
                            "only works with associated array active")
                      )
    parser.add_option("-o", "--additional_options", dest="additional_options",
                      default=[], action="append",
                      help=("additional options for internal "
                            "routines, should format of 'offset=10'"))
    parser.add_option("-b", "--blank_ranges", dest="blank_ranges", nargs=2, action="append", default=[], type=int,
                      help="manually added range \"-b 100 200\"; blank channels 100 to 200")
    try:
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "DESPIKE", "Invalid option")
        pyclass.sicerror()
        return
    #
    for option in options.additional_options:
        key, value = option.split("=")
        options._update_loose({key: value})
    if (not pyclass.gotgdict()):
        pyclass.get(verbose=False)
    # Store options from the command
    if options.smooth_data_before > 1:
        pyclass.comm("smooth box {0}".format(options.smooth_data_before))
    data = deepcopy(pyclass.gdict.ry.__sicdata__)
    cleaned_data, convolved_data = determine_blanks_and_fast_changes(data)
    data[np.where(convolved_data == 0.0)] = np.nan
    #
    wavelet_length = int(options.length)
    cutoff = int(options.cutoff)
    #
    if getattr(pyclass.gdict, "ry", None) is None:
        pyclass.message(
            pyclass.seve.e,
            "DESPIKE", "RY array not found, no spectra loaded"
        )
        pyclass.sicerror()
        return
    #
    if options.lookup_table is not None:
        # load lookup table
        if not os.path.exists(options.lookup_table):
            pyclass.message(
                pyclass.seve.e, "DESPIKE",
                "lookup table {0} not found".format(options.lookup_table)
            )
        if "lookup_table" not in globals():
            global lookup_table
            lookup_table = pandas.read_pickle(options.lookup_table)
        rex = re.search("SOF-(.+)_(\d+)_S",
                        str(pyclass.gdict.telescope.__sicdata__))
        if rex is None:
            pyclass.message(
                pyclass.seve.e,
                "DESPIKE", "cound parse telescope name {0}".format(
                    pyclass.gdict.telescope.__sicdata__
                )
            )
            return
        array, pix = rex.groups()
        telescope = "{0}_PX{1:02d}_S".format(array, int(pix))
        lookup = [int(pyclass.gdict.scan.__sicdata__), int(
            pyclass.gdict.subscan.__sicdata__), telescope]
        # check ON data for detected spurs
        spur_table_on = lookup_table[(lookup_table.scannum == lookup[0]) &
                                     (lookup_table.subscan == lookup[1]) &
                                     (lookup_table.backend == lookup[2])
                                     ]
        # if len(spur_table_on)>0:
        #    print "{0} found in lookup table".format(str(lookup))
        # check OFF data for detected spurs
        spur_table_off = lookup_table[(lookup_table.scannum == lookup[0]) &
                                      (lookup_table.subscan == lookup[1] - 1) &
                                      (lookup_table.backend == lookup[2])
                                      ]
        # if len(spur_table_off)>0:
        #    print "{0} found in lookup table for OFF".format(str(lookup))
        # check CAL data for spurs
        spur_table_cal = lookup_table[(lookup_table.scannum == lookup[0] - 1) &
                                      (lookup_table.instmode == "CAL") &
                                      (lookup_table.backend == lookup[2])
                                      ]
        # if len(spur_table_cal)>0:
        #    print "{0} found in lookup table for CAL".format(str(lookup))
        #
        spur_table = pandas.concat(
            [spur_table_on, spur_table_off, spur_table_cal])
        if hasattr(options, "spur_width"):
            spur_width = int(options.spur_width)
        else:
            spur_width = 4
        #
        if len(spur_table) != 0:
            outliers = np.unique(
                np.concatenate(
                    spur_table[
                        'outliers_width_{0}'.format(spur_width)
                    ].values
                ).ravel()
            )
            if hasattr(options, "jitter"):
                for i in range(-int(options.jitter), int(options.jitter), 1):
                    outliers = np.append(outliers, outliers + i)
                outliers = np.unique(outliers)
            # add in offset due to cropping
            if hasattr(options, "offset"):
                outliers = outliers - int(options.offset)
                outliers = [outliers[(outliers > 0) & (
                    outliers < pyclass.gdict.channels)]]
        else:
            outliers = [[]]
    elif len(options.blank_ranges) > 0:
        outliers = []
        for ch_range in options.blank_ranges:
            outliers.extend(np.arange(min(ch_range), max(ch_range)))
        outliers = [outliers]
    else:
        series = pandas.Series(data)
        cleaned_series = pandas.Series(cleaned_data)

        cleaned_series = cleaned_series.interpolate()
        mov_mean = cleaned_series.rolling(
            window=wavelet_length, center=True).median()
        flat_series = series - mov_mean
        flat_cleaned_series = cleaned_series - mov_mean

        moving_std = flat_cleaned_series.rolling(
            window=wavelet_length, center=True).std()
        positive_outliers = np.where((flat_series < -1 * cutoff * moving_std))
        negative_outliers = np.where((flat_series > cutoff * moving_std))
        outliers = np.concatenate([positive_outliers, negative_outliers], 1)
        if hasattr(options, "jitter"):
            for i in range(-int(options.jitter), int(options.jitter), 1):
                outliers = np.append(outliers, outliers + i)
            outliers = [np.unique(outliers)]
    # convert detected spurs to original native resolution before smoothing
    outliers = np.array(outliers)
    if options.smooth_data_before > 1:
        outliers = int(options.smooth_data_before) * outliers[0]
        # fill in missing gaps between channels
        for i in range(-int(options.smooth_data_before),
                       int(options.smooth_data_before), 1):
            outliers = np.append(outliers, outliers + i)
        outliers = [np.unique(outliers)]
        pyclass.comm("get")
    #
    pyclass.message(
        pyclass.seve.i,
        "DESPIKE",
        ("Found {0} outliers in channels {1}, "
         "{2.scan}:{2.subscan}  {2.number} {2.telescope}").format(
            len(outliers[0]), sorted(outliers[0]), pyclass.gdict
        )
    )
    if options.use_associated_array:
        bad_index = np.array(
            np.where(
                abs(pyclass.gdict.ry.__sicdata__ -
                    pyclass.gdict.r.head.spe.bad) < 1e-6
            )
        )
        if len(bad_index[0]) > 0:
            pyclass.message(
                pyclass.seve.i,
                "DESPIKE",
                "Found {0} bad channels {1}, "
                "{2.scan}:{2.subscan}  {2.number} {2.telescope}".format(
                    len(bad_index), sorted(bad_index[0]), pyclass.gdict
                )
            )
            outliers = np.unique(np.concatenate(
                [outliers, bad_index], 1)).astype(int)
            print(outliers)
        #
        if getattr(pyclass.gdict.kosma, "blanked", None) is not None:
            pyclass.comm("delete /VARIABLE kosma%blanked")
        if getattr(pyclass.gdict.kosma, "line", None) is not None:
            pyclass.comm("delete /VARIABLE kosma%line")
        pyclass.comm("def int kosma%blanked /like ry /global")
        pyclass.gdict.kosma.blanked[outliers] = 100

        if getattr(pyclass.gdict.r, "assoc", None) is not None:
            for array in ['blanked', 'line', 'bad']:
                if getattr(pyclass.gdict.r.assoc, array, None) is not None:
                    if array == 'bad':
                        pyclass.gdict.kosma.blanked = (
                            pyclass.gdict.kosma.blanked +
                            pyclass.gdict.r.assoc.bad.data
                        )
                    pyclass.comm("associate {0} /delete".format(array))
        pyclass.comm("associate blanked kosma%blanked")
        pyclass.comm("associate line kosma%blanked")
        pyclass.comm("associate bad kosma%blanked /bad -1")
        #
        if options.plot:
            pyclass.comm("pen 0")
            pyclass.comm("plot")
            pyclass.comm("pen 1")
            pyclass.comm("hist rx R%ASSOC%bad%data")
    else:
        data[outliers] = np.nan
        # blank_value = -1.2345678e-10
        data[np.where(np.isnan(data))] = pyclass.gdict.r.head.spe.bad
        pyclass.gdict.ry = data
        if options.noise:
            # replacing with noise
            if "mov_mean" not in locals():
                series = pandas.Series(data)
                cleaned_series = pandas.Series(cleaned_data)
                cleaned_series = cleaned_series.interpolate()
                mov_mean = cleaned_series.rolling(
                    window=wavelet_length, center=True).median()
                flat_series = series - mov_mean
                flat_cleaned_series = cleaned_series - mov_mean
                moving_std = flat_cleaned_series.rolling(
                    window=wavelet_length, center=True).std()
            #
            for channel in outliers[0, :]:
                simulated_noise = np.random.normal(
                    mov_mean[channel], moving_std[channel], size=1)
                pyclass.gdict.ry[channel] = simulated_noise[0]
            return
            #
    return


if __name__ == "__main__":
    main()
