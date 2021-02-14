from __future__ import print_function
import copy
import subprocess
import os
import yaml
import math
import pickle
import random

import numpy as np
import numpy.ma as ma
import pandas as pd

from sklearn.decomposition import PCA, FastICA, NMF
from sklearn.preprocessing import normalize

import pyclass
import pgutils
from sicparse import OptionParser


def main():

    # Input parameters
    parser = OptionParser()
    parser.add_option("-n", "--n_components",
                      dest="number_components", nargs=1, default=5)

    try:
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "PCA", "Invalid option")
        pyclass.sicerror()
        return
    if (not pyclass.gotgdict()):
        pyclass.get(verbose=False)


    # Setup parameters
    number_components = int(options.number_components)+1

    redo_index = True
    if input_file:
        pyclass.comm('file in "{}"'.format(input_file))
        pyclass.comm('find')
        modification_times_yaml_file = ".{}_modification_times_pca.yaml".format(input_file.replace("/", "_"))
        modification_time = os.path.getmtime(input_file)

        if os.path.exists(modification_times_yaml_file):
            previous_modification_times = yaml.safe_load(
                open(modification_times_yaml_file))
        else:
            previous_modification_times = {}
        if previous_modification_times is None:
            previous_modification_times = {}
        if input_file in previous_modification_times.keys():
            previous_modification_time = previous_modification_times[
                input_file]
            if (round(previous_modification_time, 0) >= round(
                    modification_time, 0)):
                redo_index = False

        previous_modification_times[input_file] = modification_time
        with open(modification_times_yaml_file, 'w') as outfile:
            yaml.dump(previous_modification_times, outfile)

    idx_list = []
    idx_list.append(pyclass.gdict.idx.num)
    idx_list.append(pyclass.gdict.idx.teles)
    idx_list.append(pyclass.gdict.idx.scan)
    idx_list.append(pyclass.gdict.idx.subscan)
    idx_list.append(pyclass.gdict.idx.sourc)
    idx_list.append(pyclass.gdict.idx.line)
    idx_list.append(pyclass.gdict.idx.boff)
    idx_list.append(pyclass.gdict.idx.loff)
    idx_list.append(pyclass.gdict.idx.ver)
    idx_list = np.asarray(idx_list)
    df = pd.DataFrame(
        idx_list.T,
        columns=["number", "telescope", "scan", "subscan",
                 "source", "line", "boff", "loff", "version"]
    )

    pyclass.comm("set var user")
    pyclass.comm("import sofia")
    scan_group = df.groupby(["scan"])
    df["mission_id"] = None
    for name, group in scan_group:
        number = group.number.iloc[0]
        pyclass.comm("get {}".format(number))
        mission_id = copy.deepcopy(
            pyclass.gdict.r.user.sofia.mission_id.__sicdata__)
        df.loc[df["scan"] == group.scan.iloc[0],
               "mission_id"] = str(mission_id).strip()

    df.source = df.source.astype(str).str.strip()
    with open(pca_index_file, "wb") as filehandler:
        pickle.dump(df, filehandler)

main()
