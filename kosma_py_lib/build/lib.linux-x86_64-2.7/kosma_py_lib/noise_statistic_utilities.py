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

class NoResponse(Exception):
    pass


def radiometer_formula(alpha=2, tsys=None, freq_step=None,
                       time=None, tau=None, elevation=None):
    return np.sqrt(alpha) * tsys / (np.sqrt(np.abs(freq_step) * 1e6 * time)) * np.exp(
        tau / np.sin(elevation))


def set_dummy_window():
    pyclass.comm("set unit c")
    pyclass.comm("set wind 1000 1001")
    pyclass.comm("set unit v")


def calculate_chi_square(measured_sigmas, theoretical_sigmas):
    chi_sq = 0
    for i, measured_sigma in enumerate(measured_sigmas):
        chi_sq += (measured_sigma-theoretical_sigmas[i])**2/theoretical_sigmas[i]
    return chi_sq

class SpectraStatistic(object):
    def __init__(self, index=None):
        self.index = index
        if self.index is None:
            self.index = int(copy.deepcopy(pyclass.gdict.number.__sicdata__))
        pyclass.comm("get {}".format(self.index))

        self.update_details()
        self.get_initial_details()

    def get_statistics(self, factor=None):
        if len(np.unique(self.ry)) < 10:
            self.generate_all_blanks_csv()
        else:
            self.iterative_smooth(factor=factor)
            self.running_noise_evaluation()
            self.generate_csv_entry()

    def cut_blanked_channels_on_either_side(self):
        pyclass.comm("set unit c")
        self.update_details()
        first_channel = 0
        last_channel = self.channels
        if self.ry[0] == self.bad:
            for i in np.arange(1, self.channels, 1):
                if self.ry[i] != self.bad:
                    # Class counts from 1
                    first_channel = i+1
                    break

        if self.ry[-1] == self.bad:
            print("last channel blanked")
            for i in np.arange(self.channels-1, 0, -1):
                if self.ry[i] != self.bad:
                    last_channel = i
                    break
        self.first_channel = first_channel
        self.last_channel = last_channel
        self.first_velocity = self.velocity + ((self.first_channel-self.reference) * self.velo_step)
        self.last_velocity = self.velocity + ((self.last_channel-self.reference) * self.velo_step)
        unique_elements, counts = np.unique(self.ry, return_counts=True)
        number_count = dict(zip(unique_elements, counts))
        self.blanked_channels = 0
        try:
            number_blanks = number_count[self.bad]
        except KeyError:
            number_blanks = 0
        self.percentage_blank = number_blanks / self.channels
        if number_blanks > 0:
            self.blanked_channels = 1
        pyclass.comm("extract {} {} c".format(first_channel, last_channel))
        self.update_details()

    def get_initial_details(self):

        self.mission_id = copy.deepcopy(str(pyclass.gdict.r.user.sofia.mission_id.__sicdata__)).strip()
        self.aot_id = copy.deepcopy(str(pyclass.gdict.r.user.sofia.aot_id.__sicdata__)).strip()
        self.aor_id = copy.deepcopy(str(pyclass.gdict.r.user.sofia.aor_id.__sicdata__)).strip()

    def update_details(self):
        self.bad = float(copy.deepcopy(pyclass.gdict.r.head.spe.bad.__sicdata__))
        self.ry = copy.deepcopy(pyclass.gdict.ry.__sicdata__)
        self.channels = copy.deepcopy(pyclass.gdict.channels.__sicdata__)
        self.freq_step = copy.deepcopy(pyclass.gdict.freq_step.__sicdata__)
        self.velo_step = copy.deepcopy(pyclass.gdict.velo_step.__sicdata__)
        self.time = copy.deepcopy(pyclass.gdict.time.__sicdata__)
        self.elevation = copy.deepcopy(pyclass.gdict.elevation.__sicdata__)
        self.scan = copy.deepcopy(pyclass.gdict.scan.__sicdata__)
        self.subscan = copy.deepcopy(pyclass.gdict.subscan.__sicdata__)
        self.telescope = copy.deepcopy(str(pyclass.gdict.telescope.__sicdata__)).strip()
        self.reference = copy.deepcopy(pyclass.gdict.reference.__sicdata__)
        self.velocity = copy.deepcopy(pyclass.gdict.velocity.__sicdata__)
        self.time = copy.deepcopy(pyclass.gdict.time.__sicdata__)
        self.line = copy.deepcopy(str(pyclass.gdict.line.__sicdata__)).strip()
        self.source = copy.deepcopy(str(pyclass.gdict.source.__sicdata__)).strip()
        # self.class_tau = copy.deepcopy(str(pyclass.gdict.tau.__sicdata__)).strip()
        # self.class_tsys = copy.deepcopy(str(pyclass.gdict.tsys.__sicdata__)).strip()

    def smooth(self, factor=2):
        pyclass.comm("smooth box {}".format(factor))
        self.update_details()

    def iterative_smooth(self, factor=2):
        """ Compare Noise in a spectrum with the radiometer formula
        """
        pyclass.comm("get {}".format(self.index))
        self.cut_blanked_channels_on_either_side()
        self.measured_sigmas = []
        self.reference_sigmas = []
        self.calculated_sigmas = []
        self.freq_steps = []

        set_dummy_window()
        self.cut_blanked_channels_on_either_side()
        iterations = int(
            math.floor(
                np.log(self.channels) / np.log(factor)
            ) - 1
        )

        for i in np.arange(0, iterations, 1):
            self.update_details()
            if i != 0:
                self.smooth(factor=factor)
            pyclass.comm("base 0")
            mean_tsys = np.mean(pyclass.gdict.r.assoc.tsys.data.__sicdata__)
            mean_tau = np.mean(pyclass.gdict.r.assoc.tau.data.__sicdata__)
            theoretical_sigma = radiometer_formula(
                alpha=2,
                tsys=mean_tsys,
                tau=mean_tau,
                freq_step=pyclass.gdict.freq_step.__sicdata__,
                time=pyclass.gdict.time.__sicdata__,
                elevation=pyclass.gdict.elevation.__sicdata__
            )
            if i != 0:
                self.reference_sigmas += [self.reference_sigmas[i-1] / np.sqrt(factor)]
            else:
                self.reference_sigmas += [copy.deepcopy(pyclass.gdict.sigma.__sicdata__)]
            self.measured_sigmas += [copy.deepcopy(pyclass.gdict.sigma.__sicdata__)]
            self.calculated_sigmas += [theoretical_sigma]
            self.freq_steps += [copy.deepcopy(pyclass.gdict.freq_step.__sicdata__)]

        self.iterative_smooth_chisq = stats.chisquare(self.measured_sigmas, self.reference_sigmas).statistic

    def running_noise_evaluation(self, avg_channels=40, alpha=2):
        pyclass.comm("get {}".format(self.index))
        self.cut_blanked_channels_on_either_side()
        set_dummy_window()
        pyclass.comm("base 0")
        chunks = int(
            np.floor(
                self.channels / avg_channels
            )
        )
        run_avg = []
        end_channel = 0

        # Calculate meean_tau and mean_tsys
        mean_tau = np.mean(pyclass.gdict.r.assoc.tau.data.__sicdata__)
        mean_tsys = np.mean(pyclass.gdict.r.assoc.tsys.data.__sicdata__)

        self.xy = copy.deepcopy(self.ry)
        print(len(self.xy), self.channels)
        for i in np.arange(chunks):
            start_channel = end_channel
            end_channel = start_channel + avg_channels
            if end_channel > self.channels:
                end_channel = self.channels
            if i == chunks-1:
                #last chunk
                if end_channel <= self.channels:
                    end_channel = self.channels

            this_run_avg = np.std(pyclass.gdict.ry[start_channel:end_channel])
            this_tau = np.mean(pyclass.gdict.r.assoc.tau.data.__sicdata__[start_channel:end_channel])
            this_tsys = np.mean(pyclass.gdict.r.assoc.tsys.data.__sicdata__[start_channel:end_channel])

            theoretical_sigma = radiometer_formula(
                alpha=alpha,
                tsys=this_tsys,
                freq_step=self.freq_step,
                time=self.time,
                tau=this_tau,
                elevation=self.elevation
            )
            this_run_avg = this_run_avg / theoretical_sigma
            run_avg += [this_run_avg]
            self.xy[start_channel:end_channel] = this_run_avg

        self.running_statistic = np.sum(run_avg) / chunks
        self.mean_tsys = mean_tsys
        self.mean_tau = mean_tau
        try:
            pyclass.comm("del /var xy")
        except pgutils.PygildasError:
            pass
        pyclass.comm("def real xy /like ry")

        pyclass.gdict.ry = self.xy


        #pyclass.comm("plot")
        #print(self.running_statistic)


    def generate_csv_entry(self):
        self.csv_entry = {}
        self.csv_entry["number"] = self.index
        self.csv_entry["mission_id"] = self.mission_id
        self.csv_entry["source"] = self.source
        self.csv_entry["line"] = self.line
        self.csv_entry["scan"] = self.scan
        self.csv_entry["subscan"] = self.subscan
        self.csv_entry["telescope"] = self.telescope
        self.csv_entry["aot_id"] = self.aot_id
        self.csv_entry["aor_id"] = self.aor_id
        self.csv_entry["smooth_down_chisq"] = self.iterative_smooth_chisq
        self.csv_entry["measured_sigmas"] = ",".join([str(item) for item in self.measured_sigmas])
        self.csv_entry["calculated_sigmas"] = ",".join([str(item) for item in self.calculated_sigmas])
        self.csv_entry["reference_sigmas"] = ",".join([str(item) for item in self.reference_sigmas])
        self.csv_entry["freq_steps"] = ",".join([str(item) for item in self.freq_steps])
        self.csv_entry["sigma"] = self.measured_sigmas[0]
        self.csv_entry["run_avg_diff"] = self.running_statistic
        # self.csv_entry["class_tsys"] = self.class_tsys
        # self.csv_entry["class_tau"] = self.class_tau
        self.csv_entry["mean_tsys"] = self.mean_tsys
        self.csv_entry["mean_tau"] = self.mean_tau
        self.csv_entry["first_channel"] = self.first_channel
        self.csv_entry["last_channel"] = self.last_channel
        self.csv_entry["first_velocity"] = self.first_velocity
        self.csv_entry["last_velocity"] = self.last_velocity
        self.csv_entry["percentage_blank"] = self.percentage_blank
        self.csv_entry["blanked_channels"] = self.blanked_channels
        self.csv_entry["time"] = self.time

    def generate_all_blanks_csv(self):
        self.csv_entry = {}
        self.csv_entry["number"] = self.index
        self.csv_entry["mission_id"] = self.mission_id
        self.csv_entry["source"] = self.source
        self.csv_entry["line"] = self.line
        self.csv_entry["scan"] = self.scan
        self.csv_entry["subscan"] = self.subscan
        self.csv_entry["telescope"] = self.telescope
        self.csv_entry["aot_id"] = self.aot_id
        self.csv_entry["aor_id"] = self.aor_id
        self.csv_entry["smooth_down_chisq"] = None
        self.csv_entry["measured_sigmas"] = None
        self.csv_entry["calculated_sigmas"] = None
        self.csv_entry["reference_sigmas"] = None
        self.csv_entry["freq_steps"] = None
        self.csv_entry["sigma"] = None
        self.csv_entry["run_avg_diff"] = None
        # self.csv_entry["class_tsys"] = None
        # self.csv_entry["class_tau"] = None
        self.csv_entry["mean_tsys"] = None
        self.csv_entry["mean_tau"] = None
        self.csv_entry["first_channel"] = None
        self.csv_entry["last_channel"] = None
        self.csv_entry["first_velocity"] = None
        self.csv_entry["last_velocity"] = None
        self.csv_entry["percentage_blank"] = None
        self.csv_entry["blanked_channels"] = 1
        self.csv_entry["time"] = None
