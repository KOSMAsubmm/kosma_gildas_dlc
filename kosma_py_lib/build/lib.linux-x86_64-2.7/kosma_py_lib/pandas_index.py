import os
import pickle

import numpy as np
import pandas as pd
import yaml

import re

import pyclass
#from online_reduction.pipeline import PandasClass
from sicparse import OptionParser
import logging
from jinja2 import Environment, PackageLoader
import subprocess

def jinja_raise(msg):
    raise Exception(msg)

def debug(text):
    return ''

ENV = Environment(
    loader=PackageLoader('kosma_py_lib', 'templates'),
    trim_blocks=True,
    lstrip_blocks=True)
ENV.globals['jinja_raise'] = jinja_raise
ENV.filters['debug'] = debug


class captureTTY:
    '''
    Needed to capture output from the fortran code run with pyclass
    Class to capture the terminal content. It is necessary when you want to
    grab the output from a module created using f2py.

    Taken from StackOverflow QUestion:
    http://stackoverflow.com/questions/10803579/copy-fortran-called-via-f2py-output-in-python
    '''
    def __init__(self,  tmpFile = '/tmp/out.tmp.dat'):
        '''
        Set everything up
        '''
        self.tmpFile = tmpFile
        self.ttyData = []
        self.outfile = False
        self.save = False
    def start(self):
        '''
        Start grabbing TTY data.
        '''
        # open outputfile
        self.outfile = os.open(self.tmpFile, os.O_RDWR|os.O_CREAT)
        # save the current file descriptor
        self.save = os.dup(1)
        # put outfile on 1
        os.dup2(self.outfile, 1)
        return
    def stop(self):
        '''
        Stop recording TTY data
        '''
        if not self.save:
            # Probably not started
            return
        # restore the standard output file descriptor
        os.dup2(self.save, 1)
        # parse temporary file
        self.ttyData = open(self.tmpFile, ).readlines()
        # close the output file
        os.close(self.outfile)
        # delete temporary file
        os.remove(self.tmpFile)


def get_class_windows():
    #
    lower_range = pyclass.gdict.set.las.wind1.__sicdata__
    upper_range = pyclass.gdict.set.las.wind2.__sicdata__
    #
    windows = [window for window in zip(lower_range, upper_range) if window != (0.0,0.0)]
    return windows


def get_index(observatory="", additional_meta_parameters = [], force_reload=False, get_rms=False, get_stats=False, get_fft_power = []):

    array = []

    folder_context = {}
    folder_context["script_dir"] = "/tmp/"
    folder_context["memory_dir"] = "./"
    config = {}
    config["general"] = {}
    config["general"]["observatory"] = observatory
    config["pipeline"]= {}
    config["pipeline"]["global"] = {}
    config["pipeline"]["global"]["exclude"] = {}
    config["pipeline"]["global"]["include"] = {}
    config["pipeline"]["class_processing"] = {}
    config["pipeline"]["class_processing"] = {}
    config["pipeline"]["class_processing"]["batch"] = True
    config["pipeline"]["class_processing"]["silent"] = True

    # Now read the file name that is not stored in an easy accesible variable
    # from the show all command
    grabber = captureTTY()
    grabber.start()
    pyclass.comm("show file")
    grabber.stop()
    #
    #input_file = grabber.ttyData[0].split(":")[1].split("[")[0].strip()
    rex = re.match("Input file:\s(.+)\s+\[.+", grabber.ttyData[0])
    if rex is None:
        pyclass.message(pyclass.seve.i, "PD_INDEX", "couldn't parse filenae {0}".format(grabber.ttyData[0]))
        return
    else:
        input_file, = rex.groups()
        input_file = input_file.replace(" ","")    
    input_file = os.path.abspath(input_file)
    pyclass.message(pyclass.seve.i, "PD_INDEX", "input file {0}".format(input_file))
    if os.path.dirname(input_file)=="":
        output_path="./"
    else:
        output_path=os.path.dirname(input_file)
    pyclass.message(pyclass.seve.i, "PD_INDEX", "output path {0}".format(output_path))
    folder_context["memory_dir"] = output_path
    #
    modification_times_yaml_file = "{}/.modification_times.yml".format(
        folder_context["memory_dir"]
    )
    #
    if os.path.exists(modification_times_yaml_file):
        previous_modification_times = yaml.safe_load(open(modification_times_yaml_file))
    else:
        previous_modification_times = {}
    #
    if input_file.strip()=="none":
        pyclass.message(pyclass.seve.e, "PD_INDEX", "No file loaded".format(input_file))
        return
    elif os.path.exists(input_file):
        modification_time = os.path.getmtime(input_file)
    else:
        pyclass.message(pyclass.seve.e, "PD_INDEX", "{0} file not found".format(input_file))
        return
    #
    pyclass.message(pyclass.seve.i, "PD_INDEX", "Loading index from {} into pandas".format(input_file))
    new = True
    print previous_modification_times
    if os.path.basename(input_file) not in previous_modification_times.keys():
        previous_modification_times[os.path.basename(input_file)] = 0.0
    #return
    #try:
    if True:
        if ( (force_reload == False) and (round(previous_modification_times[os.path.basename(input_file)],0) >= round(modification_time,0))):
            pyclass.message(pyclass.seve.i, "PD_INDEX", "Loading from already created local table")
            df = pickle.load(open("{1}/.pd_index_{0}.pkl".format(os.path.basename(input_file),
                                                                 output_path), "r"))
            new = False
        else:
            pyclass.message(pyclass.seve.i, "PD_INDEX", "Generating table from class")
            pd = PandasClass(input_file=input_file, folder_context=folder_context,
                             config=config,reduction_parameters={},
                             additional_meta_parameters=additional_meta_parameters, 
                             get_rms=get_rms,
                             get_stats=get_stats,
                             get_fft_power=get_fft_power
                             )
            df = pd.df
    #except KeyError:
    #    pd = PandasClass(input_file=input_file, folder_context=folder_context,
    #                     config=config,reduction_parameters={},
    #                     additional_meta_parameters=additional_meta_parameters, get_rms=get_rms, get_stats=get_stats)
    #    df = pd.df
    #
    pyclass.message(pyclass.seve.i, "PD_INDEX", "Pandas index loaded successfully")
    #
    #
    if new:
        pickle.dump(pd.df, open("{1}/.pd_index_{0}.pkl".format(os.path.basename(input_file),
                                                                 output_path), "wb"))
    previous_modification_times[os.path.basename(input_file)] = modification_time
    with open(modification_times_yaml_file, 'w') as outfile:
        yaml.dump(previous_modification_times, outfile)
    #
    return df



class DataFrameEmpty(Exception):
    ''' Raised when the no scans are found. '''

    def __init__(self):
        self.message = "The filtered DataFrame is empty"

class ClassExecution(object):
    def run_class_script(self, script):
        # self.log.debug("In class execution")
        if hasattr(self,"config"):
            try:
                batch_processing = self.config["pipeline"]["class_processing"]["batch"]
            except KeyError:
                batch_processing = False
        else:
            batch_processing = True        
        try:
            if batch_processing:
                cmd = "class -nw @{} > /dev/null 2>&1".format(script)
                return_code = subprocess.check_call(cmd, shell=True)
            else:
                return_code = subprocess.check_call(
                    ["class", "-nw", "@{}".format(script)])
            #self.log.debug(return_code)
        except KeyboardInterrupt:
            raise SystemExit("Killed class process")


class PandasClass(ClassExecution):
    def __init__(self,
                 input_file=None,
                 folder_context=None,
                 config=None,
                 read_user_section=True,
                 reduction_parameters=None,
                 extend_dataframe=None,
                 additional_meta_parameters=None,
                 get_rms=False,
                 get_stats=False,
                 get_fft_power=[]
                 ):
        """ Represent the index of a class file in pandas
        TODO: Includes from differnt categories should be filtered with and
        """
        self.input_file = input_file
        self.folder_context = folder_context
        self.log = logging.getLogger("debug")
        self.log.debug("Entering PandasClass")
        self.get_stats = get_stats
        self.get_rms = get_rms
        self.get_fft_power = get_fft_power
        if config is None:
            raise SystemExit(
                "Aborting. PandasClass needs a "
                "loaded config file to work from."
            )
        self.config = config
        if not input_file:
            raise SystemExit(
                "Aborting. PandasClass needs an input file to work with.")
        self.input_file = input_file
        if not os.path.exists(self.input_file):
            raise SystemExit(
                (
                    "{0} doesn't exist check filename"
                ).format(self.input_file))

        last_number = None
        if extend_dataframe is not None and not extend_dataframe.empty:
            extend_dataframe_this_file = extend_dataframe[
                extend_dataframe.input_file == input_file]
            last_number = extend_dataframe_this_file.number.max()
            if np.isnan(last_number):
                last_number = None
        template = ENV.get_template("pandas.class")
        context = {
            "input_file": input_file,
            "memory_dir": folder_context["memory_dir"]
        }
        if last_number:
            context["start_from_number"] = last_number + 1

        script = ''.join(template.generate(context))
        script_name = "{}/pandas.class".format(folder_context["script_dir"])
        with open(script_name, "w") as script_file:
            script_file.write(script)

        self.run_class_script(script_name)
        index_file = "{}/idx.csv".format(folder_context["memory_dir"])
        standard_index = [
            "index","number","version","telescope", "scan", "subscan", "boff", "loff", "line", "source"
        ]
        # first column number is taken as index
        self.df = pd.read_csv(index_file, names=standard_index)
        #
        if self.df.empty:
            raise DataFrameEmpty
        # For SOFIA/GREAT observations we can expand the data frame to also
        # include items from the SOFIA user sections
        try:
            observatory = config["general"]["observatory"]
        except KeyError:
            observatory = "SOFIA_GREAT"
        if observatory == "SOFIA_GREAT" and read_user_section:

            scans = self.df["scan"].unique()
            template = ENV.get_template("pandas_aot_aor.class")
            context = {
                "input_file": input_file,
                "memory_dir": folder_context["memory_dir"],
                "scans": scans
            }
            script = ''.join(template.generate(context))

            script_name = "{}/get_aot_aor.class".format(
                folder_context["script_dir"])
            with open(script_name, "w") as script_file:
                script_file.write(script)
            self.run_class_script(script_name)
            index_file = "{}/scan_aot_aor.csv".format(
                folder_context["memory_dir"])

            self.aot_df = pd.read_csv(
                index_file,
                names=[
                    "scan", "mission_id", "aot_id", "aor_id", "posangle",
                    "utobs", "cdobs"
                ])
            if len(self.aot_df)==0:
                self.log.error("no user section data found")
            else:
                self.df = pd.merge(self.df, self.aot_df, on="scan")
                # Create a new column that contains the tile number (if any)
                dummy_df = self.df.ix[:, "utobs":"cdobs"]
                dummy_df["ut_time"] = dummy_df["utobs"] / (2. * np.pi) * 24
                dummy_df["ut_time"] = dummy_df["ut_time"].map(float)
                dummy_df["residual_hour"] = dummy_df["ut_time"].mod(1)
                dummy_df["ut_hour"] = (
                    dummy_df["ut_time"] - dummy_df["residual_hour"]).map(int)
                dummy_df["ut_minute"] = dummy_df["residual_hour"] * 60
                dummy_df["residual_minute"] = dummy_df["residual_hour"].mod(60)
                dummy_df["ut_minute"] = (
                    dummy_df["ut_minute"] - dummy_df["residual_minute"]).map(int)
                dummy_df["ut_second"] = dummy_df["residual_minute"] * 60
                dummy_df["ut_year"] = dummy_df["cdobs"].apply(
                    lambda x: x.split("-")[2].strip())
                dummy_df["ut_month"] = dummy_df["cdobs"].apply(
                    lambda x: x.split("-")[1].upper().strip())
                dummy_df["ut_day"] = dummy_df["cdobs"].apply(
                    lambda x: x.split("-")[0].strip())
                self.df["timestamp_str"] = dummy_df[
                    ["ut_year", "ut_month", "ut_day",
                     "ut_hour", "ut_minute", "ut_second"]
                ].apply(lambda x: "{}-{}-{}T{}:{}:{:8.6f}".format(
                    x[0], x[1], x[2], x[3], x[4], x[5]), axis=1
                )

                self.df['timestamp'] = pd.to_datetime(
                    self.df["timestamp_str"], format='%Y-%b-%dT%H:%M:%S.%f')
        # append additional parameters
        if additional_meta_parameters is not None:
            self.log.debug("Collecting additional parameters PandasClass")
            parameters_to_collect = ['number','R%HEAD%GEN%VER']
            #
            drop_columns = [column for column in ["cdobs", "utobs"] if column in self.df.columns]
            if len(drop_columns)>0:
                self.df = self.df.drop(drop_columns, 1)
            # check parameters are not already in dataframe
            for additional_meta_parameter in additional_meta_parameters:
                if additional_meta_parameter is None:
                    continue
                if additional_meta_parameter.split("%")[-1] not in self.df.columns:
                    parameters_to_collect.append(additional_meta_parameter)
            #parameters_to_collect.extend(["R%HEAD%GEN%CDOBS", "utobs"])
            # generate class script from template
            template = ENV.get_template("pandas_additional_meta.class")
            context = {
                "input_file": self.input_file,
                "memory_dir": self.folder_context["memory_dir"],
                "meta_params":
                ["\'{}\'".format(param) for param in parameters_to_collect],
                "print_info_on_number": 1000,
            }
            #
            script = ''.join(template.generate(context))
            script_name = "{}/pandas_additional_meta.class".format(
                self.folder_context["script_dir"])
            with open(script_name, "w") as script_file:
                script_file.write(script)
            self.run_class_script(script_name)
            index_file = "{}/pandas_additional_meta.csv".format(
                folder_context["memory_dir"])
            #
            parameters_to_collect[parameters_to_collect.index('R%HEAD%GEN%VER')] = "version"
            self.additional_df = pd.read_csv(
                index_file,
                names=[
                    param.split("%")[-1].lower()
                    for param in parameters_to_collect
                ], delimiter="|")
            #print self.df.number,self.additional_df.number
            self.df = pd.merge(self.df, self.additional_df, on = ["number","version"])
            #print self.df, all([True for column in ["utobs", "cdobs"] if column in self.df.columns]), self.df.columns
            #print [column in self.df.columns for column in ["utobs", "cdobs"]]
            # convert utobs column to timestamp
            if  all([column in self.df.columns for column in ["utobs", "cdobs"]]):
                dummy_df = self.df.loc[:, ["utobs", "cdobs"]]
                # time
                dummy_df["ut_hour"] = (np.floor(24*dummy_df["utobs"]/(2*np.pi)))
                dummy_df["ut_minute"]  = np.floor((24*dummy_df["utobs"]/(2*np.pi) - dummy_df["ut_hour"])*60.0)
                dummy_df["ut_second"]  = ((24*dummy_df["utobs"]/(2*np.pi) - dummy_df["ut_hour"])*60 - dummy_df["ut_minute"])*60
                
                # date                
                dummy_df["ut_year"] = dummy_df["cdobs"].apply(
                    lambda x: x.split("-")[2].strip())
                dummy_df["ut_month"] = dummy_df["cdobs"].apply(
                    lambda x: x.split("-")[1].upper().strip())
                dummy_df["ut_day"] = dummy_df["cdobs"].apply(
                    lambda x: x.split("-")[0].strip())
                self.df["timestamp_str"] = dummy_df[
                    ["ut_year", "ut_month", "ut_day",
                     "ut_hour", "ut_minute", "ut_second"]
                ].apply(lambda x: "{}-{}-{}T{:002.0f}:{:002.0f}:{:8.6f}".format(
                    x[0], x[1], x[2], x[3], x[4], x[5]), axis=1
                )
                #
                self.df['timestamp'] = pd.to_datetime(
                    self.df["timestamp_str"], format='%Y-%b-%dT%H:%M:%S.%f')

            #
        if get_rms:
            self._get_rms()
        if get_stats:
            self._get_stats()
        #
        self.df.set_index('number',inplace=True)
        # For now only one forward efficiency is implemented. Reason is that
        # for all flight series on SOFIA We had a forward_efficiency of 0.97
        # TODO: Make this part of the pipeline generic
        forward_efficiency = reduction_parameters.get("forward_efficiency",
                                                      0.97)
        self.df["feff"] = forward_efficiency

        # Placeholders for the forward and beam efficiencies
        self.df["beff"] = 0

        main_beam_efficiencies = reduction_parameters.get(
            "main_beam_efficiencies", None)
        # if main_beam_efficiencies are given in the reduction_parameters add
        # them to the pandas dataframe
        if main_beam_efficiencies:
            # For every date range check the individual defined pixels and
            # update the
            for date_range in main_beam_efficiencies.keys():
                start_date, end_date = date_range.split("-")
                start_date = [int(i) for i in start_date.split(".")]
                start_date = datetime.date(start_date[2], start_date[1],
                                           start_date[0])
                start_date = start_date.strftime('%Y-%m-%d')
                end_date = [int(i) for i in end_date.split(".")]
                end_date = datetime.date(end_date[2], end_date[1], end_date[0])
                end_date = end_date.strftime('%Y-%m-%d')
                if self.df.loc[(self.df["timestamp"] >= start_date)
                               & (self.df["timestamp"] <= end_date)].empty:
                    continue
                for pixel in main_beam_efficiencies[date_range].keys():
                    if not self.df[self.df["telescope"] == pixel].empty:
                        self.df.loc[(self.df["timestamp"] >= start_date) &
                                    (self.df["timestamp"] <= end_date) &
                                    (self.df["telescope"] == pixel),
                                    "beff"] = main_beam_efficiencies[
                                        date_range][pixel]

        # Now the dataframe has to be filtered based on the exclude statements
        for key, value in config["pipeline"]["global"]["exclude"].items():
            try:
                if type(value) == str:
                    self.df = self.df[~self.df[key].str.contains(
                        value, regex=False)]
                if type(value) == list:
                    for item in value:
                        self.df = self.df[~self.df[key].str.contains(
                            item, regex=False)]
            except KeyError as e:
                # Here we check if extra keywords are to be read. If we do not
                # want to read the user section we silently ignore all keywords
                # that are not in the standard_index of class.
                if read_user_section:
                    raise e
                else:
                    if key not in standard_index:
                        continue
                    else:
                        raise e

        try:
            includes = config["pipeline"]["global"]["include"] or {}
        except KeyError:
            includes = {}

        new_df = None
        for key, value in includes.items():
            value = [str(val) for val in value]
            try:
                if new_df is None:
                    new_df = self.df[self.df[key].map(str).isin(value)]
                else:
                    new_df = pd.concat(
                        [new_df, self.df[self.df[key].map(str).isin(value)]])
            except KeyError as e:
                if read_user_section:
                    raise e
                else:
                    if key not in standard_index:
                        continue
                    else:
                        raise e

        if new_df is not None:
            self.df = new_df

        if self.df.empty:
            raise DataFrameEmpty

    def group(self, groups=["telescope", "source", "line"]):
        self.grouped = self.df.groupby(groups, as_index=False)


    def _get_stats(self):
        self.windows = get_class_windows()
        self.window_string = " ".join(["{0[0]} {0[1]}".format(window) for window in self.windows])
        if self.window_string=="":
            msg = "no windows found, skipping stats collection"
            self.log.debug(msg)
            pyclass.message(pyclass.seve.e, "PD_INDEX", msg)
            return
        template = ENV.get_template("pandas_write_stats.class")
        context = {
            "input_file": self.input_file,
            "memory_dir": self.folder_context["memory_dir"],
            "windows" : self.window_string
        }
        script = ''.join(template.generate(context))

        script_name = "{}/get_stats.class".format(
            self.folder_context["script_dir"])
        with open(script_name, "w") as script_file:
            script_file.write(script)
        self.run_class_script(script_name)
        #
        index_file = "{}/pandas_stats.csv".format(
        self.folder_context["memory_dir"])
        #
        #stats_df = pd.read_csv(
        #    index_file,
        #    names=["index","number","version",'stats_mean','stats_median','stats_max','stats_min','stats_rms','stats_sum'])
        stats_df = pd.read_csv(
            index_file)
        stats_df.columns = stats_df.columns.str.strip()
        #
        if len(stats_df > 0):
            self.df = pd.merge(self.df, stats_df, on=["index","number","version"])


    def _get_rms(self):
        #
        self.windows = get_class_windows()
        self.window_string = " ".join(["{0[0]} {0[1]}".format(window) for window in self.windows])
        if self.window_string=="":
            msg = "no windows found, skipping rms collection"
            self.log.debug(msg)
            pyclass.message(pyclass.seve.e, "PD_INDEX", msg)
            return
        self.fft_definitions = ""
        if self.get_fft_power:
            # for python fft version (slow)
            #self.fft_command = "kosma\\fft /ignore_x_limits /no_plot "
            #for fft_freq in self.get_fft_power:
            #    self.fft_command += " /fft_frequency_focus {0}".format(fft_freq)
            self.fft_definitions = "def real fft_freqs[{0}] /global\nlet fft_freqs {1}\n".format(len(self.get_fft_power)," ".join(self.get_fft_power))
        template = ENV.get_template("pandas_write_rms.class")
        context = {
            "input_file": self.input_file,
            "memory_dir": self.folder_context["memory_dir"],
            "windows" : self.window_string,
            "fft_definitions" : self.fft_definitions
        }
        script = ''.join(template.generate(context))

        script_name = "{}/get_rms.class".format(
            self.folder_context["script_dir"])
        with open(script_name, "w") as script_file:
            script_file.write(script)
        self.run_class_script(script_name)
        #
        index_file = "{}/pandas_rms.csv".format(
        self.folder_context["memory_dir"])
        #
        self.rms_df = pd.read_csv(
            index_file)
        print self.rms_df.columns
        #
        if len(self.rms_df > 0):
            self.df = pd.merge(self.df, self.rms_df, on=["index","number","version"])
