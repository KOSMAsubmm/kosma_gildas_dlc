from sicparse import OptionParser
#
import numpy as np

import matplotlib
#if options.plot != "interactive":
#    matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
from scipy.optimize import leastsq, least_squares
import time
# 
from scipy.interpolate import LSQUnivariateSpline,UnivariateSpline
import os
from scipy.optimize import leastsq
from scipy import interpolate
from datetime import datetime
import pandas as pd
# least_squares
import pyclass
# create logger
import logging
from sys import getsizeof
import pickle
#
import signal
import re
import pickle

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.tick_params(axis='both', which='minor', labelsize=12)
plt.tick_params(axis='both', which='major', labelsize=16)

def emit_colored_ansi(fn):
    def new(*args):
        levelno = args[1].levelno
        if(levelno >= 50):
            color = '\x1b[31m'  # red
        elif(levelno >= 40):
            color = '\x1b[31m'  # red
        elif(levelno >= 30):
            color = '\x1b[33m'  # yellow
        elif(levelno >= 20):
            color = '\x1b[32m'  # green
        elif(levelno >= 10):
            color = '\x1b[35m'  # pink
        else:
            color = '\x1b[0m'  # normal
        args[1].levelname = color + args[1].levelname + '\x1b[0m'  # normal
        return fn(*args)
    return new

def setup_logging(log_name="spline_fit",log_filename='/tmp/spline_fit.log',level="info"):
    #
    log_lookup = {}
    log_lookup['info'] = logging.INFO
    log_lookup['debug'] = logging.DEBUG
    log_lookup['warning'] = logging.WARNING
    log_lookup['error'] = logging.ERROR    
    #
    module_logger = logging.getLogger('spline_fit')
    # check if handlers are already present
    if len(module_logger.handlers)>=2:
        return module_logger
    #    
    module_logger.setLevel(log_lookup[level])
    # create file handler which logs even debug messages
    fh = logging.FileHandler(log_filename)
    fh.setLevel(log_lookup[level])
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(log_lookup[level])
    # create formatter and add it to the handlers
    #formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #log_format = "%(asctime)s: [%(levelname)s: %(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
    #log_format = "%(asctime)s: [%(levelname)s: %(filename)s:%(lineno)s - %(funcName)s() ] %(message)s"
    log_format = "[%(levelname)s]: %(filename)s:%(lineno)s - %(funcName)s() %(message)s"    
    formatter = logging.Formatter(log_format)

    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logging.StreamHandler.emit = emit_colored_ansi(logging.StreamHandler.emit)
    # add the handlers to the logger
    module_logger.addHandler(fh)
    module_logger.addHandler(ch)
    return module_logger


def error_func(p,x,y,spline):
    return p*spline(x)-y

def error_func_offset(p,x,y,spline):
    return p[1]+p[0]*spline(x)-y
# fit a shift in the if frequency
def error_func_shift(p,x,y,spline):
    return p[0]*spline(x+p[1])-y
    
class Spectrum():
    def __init__(self, class_env=True, kalibrate_env=False):
        self.log = logging.getLogger('spline_fit')    
        if class_env:
            self.import_class()
            self.get_class_windows()
        elif kalibrate_env:
            self.import_kalibrate()
        #
        self.generate_if_freq_axis()
        self.generate_windows_mask()
    
    def import_class(self):
        #
        self.log.debug("generating spectrum container")
        self.intensity = pyclass.gdict.ry.__sicdata__.copy()
        self.x = pyclass.gdict.rx.__sicdata__.copy()
        self.fres =  pyclass.gdict.r.head.spe.fres.__sicdata__.tolist()
        self.vres =  pyclass.gdict.r.head.spe.vres.__sicdata__.tolist()
        self.ref_channel =  pyclass.gdict.r.head.spe.rchan.__sicdata__.tolist()
        self.channels = pyclass.gdict.channels.__sicdata__.tolist()
        self.xmin = pyclass.gdict.user_xmin.__sicdata__.tolist()
        self.xmax = pyclass.gdict.user_xmax.__sicdata__.tolist()
        self.lower_axis_unit = pyclass.gdict.set.las.unit.__sicdata__.tolist()
        self.number = pyclass.gdict.number.__sicdata__.tolist()
        self.telescope = pyclass.gdict.telescope.tolist()
        self.scan = pyclass.gdict.scan.tolist()
        self.subscan = pyclass.gdict.subscan.tolist()
        self.line = pyclass.gdict.line.tolist()
        self.source = pyclass.gdict.source.tolist()              
        self.bad = pyclass.gdict.r.head.spe.bad.tolist()
        self.intensity_nan = self.intensity
        self.intensity_nan[self.intensity==self.bad]=float('nan')        
        # check if user section there
        if hasattr(pyclass.gdict.r, 'user'):
            self.mission_id = pyclass.gdict.r.user.sofia.mission_id.tolist()
            self.aot_id = pyclass.gdict.r.user.sofia.aot_id.tolist()
            self.aor_id = pyclass.gdict.r.user.sofia.aor_id.tolist()
            self.processing_steps = pyclass.gdict.r.user.sofia.processing_steps.tolist()
        else:
            self.log.warning("no user section, using defaults values for user section")
            self.mission_id = "no user section"
            self.aot_id = "no user section"
            self.aor_id = "no user section"
            self.processing_steps = "no user section"
        
    def generate_if_freq_axis(self):
        #
        self.if_freq = np.abs(np.arange(len(self.intensity))*self.fres + self.fres/2)
        
    def get_class_windows(self):
        #
        self.log.debug('getting class windows')
        lower_range = pyclass.gdict.set.las.wind1.__sicdata__
        upper_range = pyclass.gdict.set.las.wind2.__sicdata__
        self.log.debug('got class windows')        
        # 
        self.windows = [window for window in zip(lower_range, upper_range) if window != (0.0,0.0)]
        
    def generate_windows_mask(self):
        windows_index = []
        for window in self.windows:
            window_index = ((self.x > min(window)) & (self.x < max(window)))
            windows_index.append(list(window_index))
        # mask for bad channels
        windows_index.append((np.isnan(self.intensity_nan)))
        #windows_index.append((self.intensity_nan))        
        #
        self.crop_edge_if_range = 200.0
        windows_index.append(((self.if_freq<self.crop_edge_if_range) | (self.if_freq>(max(self.if_freq)-self.crop_edge_if_range))))
        
        #windows_index.append(((self.if_freq>self.crop_edge_if_range) | (self.if_freq<(max(self.if_freq)-self.crop_edge_if_range))))
        # combime masks        
        self.mask = np.column_stack(tuple(windows_index)).any(axis=1)  
        
class SplineFit:
    def __init__(self, 
                 spectrum,
                 spline_nodes=50,
                 fit_data_smoothness = 250,
                 fit_spline = False,
                 spline_catalog_filename=None, 
                 store_spline_in_archive=False,
                 scale_spline_from_archive=False,
                 plot="none",
                 search_scan_range = 5,
                 output_plots_path = "./plots",
                 fit_type = "scale_spline"
                 ):
        #            
        #
        self.crop_edge_if_range = 100.0 # ignore +/- if band edges        
        self.plot = plot
        self.spline_nodes = spline_nodes
        self.spectrum = spectrum
        self.output_path = output_plots_path
        self.fit_spline = fit_spline
        self.fit_data_smoothness =fit_data_smoothness
        self.spline_catalog_filename = spline_catalog_filename
        self.store_spline_in_archive = store_spline_in_archive
        self.scale_spline_from_archive = scale_spline_from_archive
        self.search_scan_range = search_scan_range
        self.plot_tag = options.plot_name_tag
        self.fit_type = fit_type
        self.catalog_columns = [
               "telescope",
               "scan", 
               "subscan",
               "source",
               "line",
               "mission_id",
               "aot_id",
               "aor_id",
               "processing_steps",
               "spline",
               "spline_type"
            ]                
        self.log = logging.getLogger('spline_fit')
        if not os.path.exists(os.path.abspath(os.path.dirname(spline_catalog_filename))):
            msg ="output spline folder not found {0}".format(os.path.dirname(spline_catalog_filename))
            self.log.error(msg)
            pyclass.message(pyclass.seve.e, "Spline fit", msg)
            sys.exit()
        if not isinstance(spectrum,Spectrum):
            self.log.error("Input spectrum not a spectrum class")
            return
        #
    
    def __call__(self):
        #
        self.crop_spectra()
        self.smooth_spectrum()
        self.load_catalog()
        #
        if self.scale_spline_from_archive:
            self.find_best_match_spline_from_archive()        
        elif self.fit_spline:
            try:
                self.fit_fixed_node_spline()
            except:
                self.log.error("fixed node fit failed for {0}".format(self.spectrum.number))
            #try:
            #    self.fit_variable_node_spline()
            #except:
            #    self.log.error("variable node fit failed for {0}".format(self.spectrum.number))

        #
    
    def crop_spectra(self):
        #
        self.log.debug("cropping spectra")
        #print  np.where()
        self.crop_index, = np.where((self.spectrum.if_freq<self.crop_edge_if_range) | (self.spectrum.if_freq>(max(self.spectrum.if_freq)-self.crop_edge_if_range)))
        #self.spectrum.intensity[self.crop_index] = self.spectrum.bad
                
    def smooth_spectrum(self):    
        if any(pyclass.gdict.rx):
            self.log.debug("smoothing input spectra using class smooth routine")
            if self.fit_data_smoothness == 0:
                pyclass.comm("smooth box {0}".format(self.fit_data_smoothness))
            self.intensity_smooth = pyclass.gdict.ry.copy()
            smooth_fres = pyclass.gdict.r.head.spe.fres.__sicdata__.tolist()
            self.if_freq_smooth = np.abs(np.arange(len(self.intensity_smooth))*smooth_fres + smooth_fres/2)
            self.x_smooth =  pyclass.gdict.rx.copy()
            pyclass.comm("get number")            
        else:
            self.log.debug("smoothing input spectra using numpy smooth routine")        
            self.if_freq_smooth, self.intensity_smooth  = smooth(self.spectrum.if_freq,
                                                         self.spectrum.intensity,
                                                         self.fit_data_smoothness,
                                                         True,
                                                         self.spectrum.bad)
            self.x_smooth, nan = smooth(self.spectrum.x,
                                    self.spectrum.intensity,
                                    self.fit_data_smoothness,
                                    True,
                                    self.spectrum.bad)                                                     
                                    
    #
    def generate_flat_spline(self):
        channel_nb = 4000
        steps=100
        edge = 100
        self.spline = LSQUnivariateSpline(x = np.arange(channel_nb),
                                          y = np.ones(channel_nb),
                                          t = np.arange(edge,channel_nb-edge,steps),
                                          k = 3)
        return self.store_spline_in_catalog(self.spline, spline_type="flat",return_row_only=True)
            
    def fit_fixed_node_spline(self):    
        self.log.info("starting fit for spectra number: {0.spectrum.number},"\
                      "spline nodes: {0.spline_nodes}, "\
                      "smooth box: {0.fit_data_smoothness}".format(self))        
        node_channels = np.arange(3,self.spline_nodes-3)*\
                           ((max(self.spectrum.if_freq)-min(self.spectrum.if_freq))/self.spline_nodes)
        weight = []
        weight.append(self.intensity_smooth==self.spectrum.bad)
        weight.append(np.isnan(self.intensity_smooth))
        w = np.column_stack(tuple(weight)).any(axis=1)
        # check for n
        #
        # selected valid nodes positions
        valid_nodes = np.where((np.min(self.if_freq_smooth[~w]) < node_channels) &  (node_channels < np.max(self.if_freq_smooth[~w])))
        #        
        self.spline_fixed = LSQUnivariateSpline(x=self.if_freq_smooth[~w],
                                                y=self.intensity_smooth[~w],
                                                t=node_channels[valid_nodes],
                                                k=3)
        #t=node_channels[valid_nodes],k=3)                  
        self.spline_fit_fixed = self.spline_fixed(self.spectrum.if_freq)     
        self.spline_fit_fixed[self.crop_index] = self.spectrum.bad
        #
        # 
        self.spectrum.best_fit_spline = self.spline_fit_fixed
        self.spectrum.corrected_intensity =  self.spectrum.intensity_nan - self.spectrum.best_fit_spline
        #        
        if self.store_spline_in_archive:
           self.store_spline_in_catalog(self.spline_fixed, spline_type="fixed_grid")
        
    def fit_variable_node_spline(self):
        self.log.info("starting variable spline fit for spectra number: {0.spectrum.number}".format(self))            
        w = np.isnan(self.intensity_smooth)
        bad = self.spectrum.intensity==self.spectrum.bad
        self.spline_variable = UnivariateSpline(self.if_freq_smooth[~w],
                                                self.intensity_smooth[~w])
        self.spline_variable.set_smoothing_factor(0.1)
        #
        self.spline_fit_variable = self.spline_variable(self.spectrum.if_freq)        
        self.spline_fit_variable[self.crop_index] = self.spectrum.bad
        self.spectrum.best_fit_spline = self.spline_fit_variable
        self.spectrum.corrected_intensity =  self.spectrum.intensity_nan - self.spectrum.best_fit_spline
        #
        if self.store_spline_in_archive:
           self.store_spline_in_catalog(self.spline_fixed, spline_type="variable_grid")    

        
    def plot_spline_fit(self,fit,show=True,save="save_best"):
        if type(fit) is pd.core.frame.DataFrame:
            fit = fit.iloc[-1,:]
            self.log.warning("two entries found for index: {0.name} plot last one".format(fit))
        self.best_fit = fit
        spline = fit.spline
        scale_factor = fit.scale_factor
        shift_factor = fit.shift_factor
        shift_offset = fit.shift_offset
        #
        #fig,(ax_data,ax_res)  = plt.subplots(2,1,figsize=(15,7))
        fig = plt.figure(figsize=(10,5))
        ax_data = fig.add_subplot(211)
        ax_res = fig.add_subplot(212)
                
        bad_data = self.spectrum.intensity==self.spectrum.bad
        #     
        ax_data.plot(self.spectrum.x, self.spectrum.intensity_nan,lw=1, label="data")
        ax_data.plot(self.spectrum.x[~self.spectrum.mask], self.spectrum.intensity_nan[~self.spectrum.mask], 'g',lw=1, label="data used for fit")
        nan,smooth_spectra = smooth(self.spectrum.x,self.spectrum.intensity_nan,self.fit_data_smoothness,False,self.spectrum.bad)
        ax_data.plot(self.spectrum.x, smooth_spectra, 'b',lw=1, label="")
        #
        ax_data.plot(self.spectrum.x,  shift_offset + scale_factor*spline(self.spectrum.if_freq + shift_factor), 
                      'r', lw=1, label="scaled spline ({0:3.2f})".format(scale_factor) )
        ax_data.plot(self.spectrum.x,  spline(self.spectrum.if_freq), 'r--', lw=1, label="master spline"  ) 
        #
        index = []
        for knot in spline.get_knots():
            index.append(find_index_nearest(self.spectrum.if_freq,knot))
        ax_data.plot(self.spectrum.x[index], shift_offset + scale_factor*spline(spline.get_knots()), 'ro')
        ax_data.grid()
        ax_data.legend(loc="upper left", ncol=2)        
        # residual
        residual = self.spectrum.intensity - shift_offset - scale_factor*spline(self.spectrum.if_freq+shift_factor)
        nan,smooth_residual = smooth(self.spectrum.x,residual,50,False,self.spectrum.bad)
        ax_res.plot(self.spectrum.x[~bad_data],  residual[~bad_data], lw=1, label="residual")
        ax_res.plot(self.spectrum.x,  smooth_residual, lw=1, label="smoothed residual")
        ax_res.grid()
        ax_res.legend(loc="upper left", ncol=2)
        ax_res.set_xlabel("V$_{LSR}$ ($\mbox{km\,s}^{-1}$)")
        ax_res.set_ylabel("T$_{mb}$ (K)")
        ax_data.set_ylabel("T$_{mb}$ (K)")
        #ax_data.set_xlabel("Velocity (km/s)")
        #
        xmin = pyclass.gdict.user_xmin.__sicdata__
        xmax = pyclass.gdict.user_xmax.__sicdata__
        ymin = pyclass.gdict.user_ymin.__sicdata__
        ymax = pyclass.gdict.user_ymax.__sicdata__
        #print ymax-ymin
        if (ymax-ymin!=1.0):
            ax_data.set_ylim((ymin, ymax))
            ax_res.set_ylim((ymin, ymax))
        title = "{1.spectrum.number};{1.spectrum.scan}:{1.spectrum.subscan}, scale-factor: {0.scale_factor:3.3f}, shift-factor: {0.shift_factor:3.3f}, offset-factor: {0.shift_offset:3.3f} \n chi-squared:{0.chi_squared:3.3f} {0.name}".format(fit,self)
        title = title.replace("_","\_")
        fig.suptitle(title)
        fig.tight_layout()
        #
        if show==True:
            plt.show(block=False)
        if "save_best" in save:
            filename = "{0.output_path}/{0.spectrum.number}_{0.spectrum.scan}_{0.spectrum.subscan}_{0.spectrum.telescope}_{0.fit_type}_sr_{0.search_scan_range}{0.plot_tag}".format(self)            
            self.log.info("saving {0}.png".format(filename))
            # 
            if "with_pickle" in self.plot:
                fig.tight_layout()
                fig.suptitle("")
                self.log.info("saving {0}.pkl".format(filename))
                with open(filename+".pkl", 'wb') as f:
                    pickle.dump(fig, f)
            fig.savefig(filename+".png")
            plt.close(fig)
        
    def find_best_match_spline_from_archive(self):
        #
        self.log.info("finding best match spline from archive for {0.spectrum.number} within {0.search_scan_range} scan numbers of {0.spectrum.number}".format(self))        
        if self.spline_catalog is None:
            self.log.error("No spline archive loaded, exiting")
            sys.exit(2)
        self.fit_catalog =  self.spline_catalog.copy()

        #                                   ]
        self.fit_catalog = self.fit_catalog[((self.fit_catalog.scan > self.spectrum.scan-self.search_scan_range) & (self.fit_catalog.scan < self.spectrum.scan+self.search_scan_range)) &
                                            (self.fit_catalog.telescope == self.spectrum.telescope) &
                                            (self.fit_catalog.spline_type == "fixed_grid")
                                           ]        
        self.fit_catalog = pd.concat([self.fit_catalog,self.generate_flat_spline()])
        
        self.fit_catalog['scale_factor'] = np.zeros(len(self.fit_catalog))
        self.fit_catalog['shift_factor'] = np.zeros(len(self.fit_catalog))        
        self.fit_catalog['shift_offset'] = np.zeros(len(self.fit_catalog))          
        self.fit_catalog['chi_squared'] = np.zeros(len(self.fit_catalog))
        self.fit_catalog['rms'] = np.zeros(len(self.fit_catalog))
        self.fit_catalog['std'] = np.zeros(len(self.fit_catalog))
        #print self.fit_catalog.columns
        # check for duplicates, some duplicates in the catalog, mess up best fit selection
        self.fit_catalog = self.fit_catalog.drop_duplicates(subset=self.index_parameters,keep="first")
        #        
        #
        x = self.spectrum.if_freq
        y = self.spectrum.intensity_nan
        self.log.debug("looping over {0} splines".format(len(self.fit_catalog)))
        if self.fit_type == "scale_spline_fixed_shift":
            shift_factors = [0.0,100.0]
        else:
            shift_factors = [0.0]
        fitted_splines = []
        for shift_factor in shift_factors:
            for i,(index,row) in enumerate(self.fit_catalog.iterrows()):
                name = row.name
                self.log.debug("fitting ({0}/{1}):{2}".format(i,len(self.fit_catalog),index))
                spline = row.spline
                try:
                    if self.fit_type == "scale_spline":
                        popt = least_squares(error_func,x0=[0.0],args=(x[~self.spectrum.mask],y[~self.spectrum.mask],row.spline))
                        shift_factor = 0.0
                        shift_offset = 0.0
                    elif self.fit_type == "scale_spline_and_offset":
                        popt = least_squares(error_func_offset,x0=[0.0,-200.0],args=(x[~self.spectrum.mask],y[~self.spectrum.mask],row.spline))
                        shift_factor = 0.0
                    elif self.fit_type == "scale_spline_and_shift":
                        popt = least_squares(error_func_shift,x0=[0.0,100.0],args=(x[~self.spectrum.mask],y[~self.spectrum.mask],row.spline))
                    elif self.fit_type == "scale_spline_fixed_shift":
                        self.log.debug("fitted with {0} shift_factor".format(shift_factor))                  
                        popt = least_squares(error_func,x0=[0.0],args=(x[~self.spectrum.mask]+shift_factor,y[~self.spectrum.mask],row.spline))
                except Exception as e:
                    self.log.error(e)
                    continue
                if len(popt.x) == 1:
                    scale_factor = popt.x[0]
                    self.log.debug("fitted with {0} scale_factor".format(scale_factor))
                    shift_offset = 0.0
                elif self.fit_type == "scale_spline_and_offset":
                    scale_factor = popt.x[0]            
                    shift_factor = 0.0
                    shift_offset = popt.x[1]
                    self.log.debug("fitted with {0} shift_factor".format(shift_factor))
                    self.log.debug("fitted with {0} shift_offset".format(shift_offset))                    
                elif len(popt.x) == 2:
                    scale_factor = popt.x[0]            
                    shift_factor = popt.x[1]
                    shift_offset = 0.0
                    self.log.debug("fitted with {0} shift_factor".format(shift_factor))
                    self.log.debug("fitted with {0} scale_factor".format(scale_factor))
                #
                #plot_fitted_spline_in_gildas(pause=False)            
                residual = self.spectrum.intensity_nan - shift_offset - scale_factor*spline(self.spectrum.if_freq + shift_factor)
                #chi_squared = np.nanmean((residual[~self.spectrum.mask])**2/(scale_factor*spline(self.spectrum.if_freq[~self.spectrum.mask]))**2)           
                chi_squared = np.nanmean((residual[~self.spectrum.mask])**2)
                nan,smooth_residual = smooth(self.spectrum.if_freq,residual,self.fit_data_smoothness,False,self.spectrum.bad)
                chi_squared = np.nanmean((smooth_residual[~self.spectrum.mask])**2)
                std = np.nanstd(residual[~self.spectrum.mask])
                rms = np.sqrt(np.nanmean(np.square(residual[~self.spectrum.mask])))
                row.scale_factor = scale_factor
                row.shift_factor = shift_factor
                row.shift_offset = shift_offset                
                row.chi_squared = chi_squared           
                row.rms = rms
                row.std = std 
                row.name = "{0}_shift_{1:3.1f}_offset_{2:3.1f}".format(name,shift_factor,shift_offset)
                #
                fitted_splines.append(row)
                #
        # add a polynomial order 1
        poly_tables = []
        if self.fit_type == 'scale_spline_and_offset':
            poly_order = 0
        else:
            poly_order = 0
        for poly_order in range(poly_order):
            poly = np.poly1d(np.polyfit(x[~self.spectrum.mask],y[~self.spectrum.mask], poly_order))
            channel_nb = 4000
            steps=100
            edge = 100
            poly_spline = LSQUnivariateSpline(x = np.arange(channel_nb),
                                              y = poly(np.arange(channel_nb)),
                                              t = np.arange(edge,channel_nb-edge,steps),
                                              k = 3)
            residual = self.spectrum.intensity_nan - poly(self.spectrum.if_freq)
            nan,smooth_residual = smooth(self.spectrum.if_freq,residual,self.fit_data_smoothness,False,self.spectrum.bad)            
            chi_squared = np.nanmean((smooth_residual[~self.spectrum.mask])**2)
            poly_table = self.store_spline_in_catalog(poly_spline, spline_type="poly_{0}".format(poly_order),return_row_only=True)
            poly_table["chi_squared"] = chi_squared
            poly_table["scale_factor"] = [1.0]
            poly_table["shift_factor"] = [0.0]
            poly_table["shift_offset"] = [0.0]
            poly_tables.append(poly_table)
        #

        #
        self.fit_catalog = pd.DataFrame(fitted_splines)
        if len(poly_tables) > 0:
            poly_table = pd.concat(poly_tables)        
            self.fit_catalog = pd.concat([self.fit_catalog,poly_table])        
        #plt.scatter(np.arange(len(self.fit_catalog)), fitted_splines.chi_squared)
        best_spline_idx = self.fit_catalog.chi_squared.idxmin()
        self.best_fit_spline = self.fit_catalog.loc[best_spline_idx,:]
        self.spectrum.best_fit_spline =  self.best_fit_spline.shift_offset + self.best_fit_spline.scale_factor*self.best_fit_spline.spline(self.spectrum.if_freq + self.best_fit_spline.shift_factor)        
        # don't add intensity offset back in for high continuum sources
        #self.spectrum.corrected_intensity =  self.spectrum.intensity_nan - self.best_fit_spline.shift_offset - self.best_fit_spline.scale_factor*self.best_fit_spline.spline(self.spectrum.if_freq + self.best_fit_spline.shift_factor)
        if ((self.fit_type == 'scale_spline_and_offset')  and (self.best_fit_spline.spline_type=='flat')):
            self.spectrum.corrected_intensity =  self.spectrum.intensity_nan
            #self.log.warning("doing nothing")
        else:
            self.spectrum.corrected_intensity =  self.spectrum.intensity_nan  - self.best_fit_spline.scale_factor*self.best_fit_spline.spline(self.spectrum.if_freq + self.best_fit_spline.shift_factor)        
        self.spectrum.corrected_intensity[np.isnan(self.spectrum.corrected_intensity)] = self.spectrum.bad
        self.spectrum.processing_steps = self.best_fit_spline.processing_steps 
        self.log.info("best fit: {0}".format(self.best_fit_spline.processing_steps))
        #        
        if self.plot=="interactive":
            fig, ax = plt.subplots()
            ax.scatter(self.fit_catalog.scan, self.fit_catalog.chi_squared, picker=True, label="spline fits")
            ax.scatter(self.fit_catalog[self.fit_catalog.spline_type=="flat"].scan,  
                       self.fit_catalog[self.fit_catalog.spline_type=="flat"].chi_squared,
                       c='r', marker="*", s = 100, label="Base 0")
            ax.scatter(self.fit_catalog[self.fit_catalog.spline_type.str.contains("poly")].scan,  
                       self.fit_catalog[self.fit_catalog.spline_type.str.contains("poly")].chi_squared,
                       c='k', marker="*", s = 100, label="Base 1")
            ax.scatter([self.fit_catalog.scan.loc[best_spline_idx]],
                       [self.fit_catalog.chi_squared.loc[best_spline_idx]],
                       c='g', marker="*", s=100, label="Best fit")                       
            ax.set_yscale("log")
            plt.axvline(x=self.spectrum.scan, linewidth=2, color='r', alpha=0.5)
            fig.suptitle("{0.scan}:{0.subscan} {0.number} {0.telescope}\n click on point to select fit".format(self.spectrum))
            fig.canvas.mpl_connect('pick_event', self.plot_onpick)
            #cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
            plt.legend()
            plt.show()
        elif "save_best" in self.plot:
            self.plot_spline_fit(self.fit_catalog.loc[best_spline_idx,:],show=False,save=self.plot)

    def plot_onpick(self,event):
        #print event.__dict__, event.ind
        for i in event.ind:
           self.plot_spline_fit(self.fit_catalog.iloc[i]) 

    def onclick(self,event):
        print('%s click: button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
              ('double' if event.dblclick else 'single', event.button,
               event.x, event.y, event.xdata, event.ydata)) 
        
    def load_catalog(self):
        if not os.path.exists(self.spline_catalog_filename):
            self.log.warning("spline catalog  {0.spline_catalog_filename} not found".format(self))
            self.log.info("starting a fresh catalog")
            #self.spline_catalog = pd.DataFrame(dict(zip(self.catalog_columns,[[] for i in range(len(self.catalog_columns))])))
            self.spline_catalog = None
        else:            
            self.log.debug("loading catalog {0.spline_catalog_filename}".format(self))
            #self.spline_catalog = pd.read_pickle(self.spline_catalog_filename)
            self.spline_catalog = pd.read_pickle(self.spline_catalog_filename)
        
    def check_if_spline_in_catalog(self, row):
        #
        return
    
    def reduce_spline_size(self,spline):    
        '''
        drop excess data from spline
        blows up catalog size        
        '''
        _data = []
        for ele in spline._data:
            if type(ele)==np.ndarray:
                _data.append(ele[:2])
            else:
                _data.append(ele)
        _data = tuple(_data)
        spline.__setattr__("_data",_data)
        
        
    def store_spline_in_catalog(self, spline,spline_type,return_row_only=False):
        row_dict = {}
        index_parameters = ["telescope", "scan", "subscan", "source", "line","spline_type","spline_nodes"]
        self.index_parameters = index_parameters
        # generate row
        for param in self.catalog_columns:
            param_value = getattr(self.spectrum,param,None)
            if param_value is not None:
                row_dict[param] = param_value
        #
        self.reduce_spline_size(spline)
        row_dict["spline"] = spline
        row_dict["spline_nodes"] = self.spline_nodes
        row_dict["spline_type"] = spline_type
        row_dict["time_added"] = datetime.now()
        # check if spline already in catalog
        index = "_".join([str(row_dict[param]).replace(" ","") for param in index_parameters])
        found = re.findall("(\d+:\d+)",self.spectrum.processing_steps)
        if len(found) == 2:
            index += "_"
            index += "_".join(found)        
        #
        self.log.debug("{0}".format(index))
        self.log.debug("adding spline to catalog")        
        row_table = pd.DataFrame(row_dict,index=[index])
        if  return_row_only:
            return row_table
        if self.spline_catalog is None:
            self.spline_catalog = row_table
        else:
            #
            if index in self.spline_catalog.index.values:
                self.log.warning("spline {0}".format(index))
                self.log.warning("already in catalog for, overwriting what was there")
                self.spline_catalog = self.spline_catalog.drop(index)
            self.spline_catalog = pd.concat([self.spline_catalog,row_table])
        self.spline_catalog.to_pickle(self.spline_catalog_filename)
        

      
        
def find_index_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx       
        
        
def smooth(x,y, box_pts, class_like=True, bad_channel="-1.23456000e+10"):
    box = np.ones(box_pts)/box_pts
    bad_index = y==bad_channel
    y_smooth = np.convolve(y[~bad_index], box, mode="same")
    #
    if class_like:
        x_smooth_grid = x[np.arange(0,len(y)-box_pts,box_pts)+int(box_pts/2)]
        interpolate_y = interpolate.griddata(points=x[~bad_index],
                                             values=y_smooth,
                                             xi=x_smooth_grid)
        #
        return x_smooth_grid,interpolate_y
    else:
        return x,y_smooth

def compare_class_python_smooth():
    x = pyclass.gdict.rx.__sicdata__
    y = pyclass.gdict.ry.__sicdata__
    # check smooth data
    fig, (ax_smooth) = plt.subplots(1,1)
    bad_index = y==pyclass.gdict.r.head.spe.bad.__sicdata__
    ax_smooth.plot(x[~bad_index],y[~bad_index], label="raw data")
    smooth_box = 100
    try:
        pyclass.comm("smooth box {0}".format(smooth_box))
    except Exception as e:
        print e
    ax_smooth.plot(pyclass.gdict.rx.__sicdata__,
                   pyclass.gdict.ry.__sicdata__,
                   'ro', ms=7, label="class smooth box {0}".format(smooth_box))
    # smooth python way
    smooth_x, smooth_y = smooth(x,y, smooth_box, 
                                class_like=True,
                                bad_channel=pyclass.gdict.r.head.spe.bad.__sicdata__)
    ax_smooth.plot(smooth_x, smooth_y, 'k+', 
                   ms=3, label="numpy smooth box {0}".format(smooth_box)
                    )
    #
    plt.legend()
    fig.show()
            

def handler(num, flame):
    log = logging.getLogger('spline_fit')
    log.info("!!ctrl+C!!")
    pyclass.message(pyclass.seve.e, "Spline fit", "Ctrl+C pressed")
    sys.exit()


def plot_fitted_spline_in_gildas(with_variable_nodes=True,pause=True):
    pyclass.comm("set unit v")
    pyclass.comm("pl")
    #pyclass.comm("clear")
    #pyclass.comm("box")
    pyclass.comm("pen 0")
    pyclass.comm("hist rx kosma%ry_orig /BLANKING 'r%head%spe%bad' 0")
    pyclass.comm("pen 1")
    pyclass.comm("hist rx kosma%spline_fit  /BLANKING 'r%head%spe%bad' 0")    
    pyclass.comm("pen 2")
    pyclass.comm("hist rx kosma%spline_corrected  /BLANKING 'r%head%spe%bad' 0")
    pyclass.comm("pen 3")    
    if pause:
        time.sleep(2)
    
    
def main():
    plot_choices = ["interactive","save_best","save_best_with_pickle","gildas_plot","none"]
    parser = OptionParser()
    parser.add_option("--fit_spline", dest="fit_spline", nargs=1, default=True)
    # number of nodes per spectra
    parser.add_option( "--spline_nodes",
                       dest="spline_nodes",
                       nargs=1,
                       type=int,
                       default=100,
                       help="number of nodes to for spline fit, default %default")
    #
    parser.add_option( "--smoothness_of_fit_data",
                       dest="smoothness_of_fit_data",
                       nargs=1,
                       type=int,
                       default=100,
                       help="number of channels to smooth data per fitting spline")
    # logging options
    parser.add_option( "--logging_level",
                       dest="logging_level",
                       nargs=1,
                       default="info",
                       choices = ["info", "debug", "warning", "error"],
                       help="set logging level %default" )
    #
    parser.add_option( "--show_plot", dest="show_plot", nargs=1, default=False,
                       help="show plot summarizing fit" )
    
    parser.add_option("--spline_archive_filename", dest="spline_archive_filename", nargs=1, default="/tmp/spline_archive.csv",
                      help="file where splines templates are stored, stored in pandas table"
                        )
    parser.add_option("--store_spline_in_archive", dest="store_spline_in_archive", action="store_true",
                      default=False, help="store fitted spline in the archive"
                        )                        
    parser.add_option("--scale_spline_from_archive", dest="scale_spline_from_archive", action="store_true",
                      default=False, help="check which spline in the archive give the best fit to the data"
                        )                        
    parser.add_option("--plot", dest="plot", default="none", help="plot fitted spline choices: {0}".format(",".join(plot_choices)), choices = plot_choices
                        )
    parser.add_option("--search_scan_range", dest="search_scan_range", type=int,
                      default=5, help="range of scan numbers to look for spline"
                        )                        
    parser.add_option("--output_plots_path", dest="output_plots_path", type=str,
                      default="./plots", help="path to save output plots to "
                        )
    parser.add_option("--plots_tag", dest="plot_name_tag", type=str,
                      default="", help="additional string to identify different reduction"
                        )
    parser.add_option("--spline_fit_type", dest="spline_fit_type",
                      default="scale_spline", choices = ["scale_spline_and_shift","scale_spline","scale_spline_fixed_shift", "scale_spline_and_offset"],
                      help="How to scale spline, either simply scale, or scale and shift frequency axis [+/- 150 MHz]"
                        )                        
    #
    signal.signal(signal.SIGINT, handler)    
    
    try:
        global options
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "Spline", "Invalid option")
        pyclass.sicerror()
        return
    #
    #
    options.output_plots_path = options.output_plots_path.replace("\"","")
    if not os.path.exists(options.output_plots_path):
        os.mkdir(options.output_plots_path)
    # sample spacing
    #
    if (not pyclass.gotgdict()):
	    pyclass.get(verbose=False)
    
	# check is there a spectrum loaded    
    if not hasattr(pyclass.gdict, "ry"):
	    pyclass.message(pyclass.seve.e, "Spline", "no spectra loaded")
	    pyclass.sicerror()
	    return
    # set logging level
    module_logger = setup_logging(log_name="spline_fit", log_filename='/tmp/spline_fit.log', level=options.logging_level)
    
    # set debug level
    log_lookup = {}
    log_lookup['info'] = logging.INFO
    log_lookup['debug'] = logging.DEBUG
    log_lookup['warning'] = logging.WARNING
    log_lookup['error'] = logging.ERROR
    module_logger.setLevel(log_lookup[options.logging_level])
    for logger_handle in module_logger.handlers:
        logger_handle.setLevel(log_lookup[options.logging_level])
	#
    #compare_class_python_smooth()
    if not hasattr(pyclass.gdict.r, 'user'):
        pyclass.comm("set var user")
        pyclass.comm("import sofia")    
        pyclass.comm("get {0}".format(pyclass.gdict.number.__sicdata__))
    if not hasattr(pyclass.gdict, 'kosma'):
        module_logging.info("creating kosma structure")
        pyclass.comm("define structure kosma /global")
    #
    if not hasattr(pyclass.gdict.kosma, "ry_orig"):
        #pyclass.comm("define structure kosma")    
        pyclass.comm("def double kosma%ry_orig /like rx")
    #
    spec = Spectrum()
    #
    # generata array
    for var in ["spline_fit","spline_corrected","ry_orig"]:      
        #print "already exists {0}".format(var), hasattr(pyclass.gdict.kosma, var)
        if not hasattr(pyclass.gdict.kosma, var):
            module_logger.info("generating kosma%{0}".format(var))    
            pyclass.comm("def double kosma%{0} /like rx".format(var))          
        if len(getattr(pyclass.gdict.kosma,var)) != pyclass.gdict.channels.__sicdata__.tolist():
            module_logger.info("deleting kosma%{0}".format(var))
            pyclass.comm("delete /var kosma%{0}".format(var))
            #time.sleep(1.5) # delete needs some time..
            pyclass.comm("def double kosma%{0} /like rx".format(var))
    setattr(pyclass.gdict.kosma, "ry_orig", spec.intensity )
    if np.count_nonzero(~np.isnan(spec.intensity))<spec.channels/2.0:
        module_logger.error("no data found")
        return
    #
    if not hasattr(pyclass.gdict.kosma, "spline_corrected"):        
        #pyclass.comm("delete /var kosma%spline_corrected")       
        pyclass.comm("def double kosma%spline_corrected /like rx /global")
        
    #return
    spl_fit = SplineFit(spectrum = spec,
                        spline_nodes = options.spline_nodes, 
                        fit_data_smoothness = options.smoothness_of_fit_data,
                        fit_spline = options.fit_spline,
                        spline_catalog_filename = options.spline_archive_filename.replace("\"",""),
                        store_spline_in_archive = options.store_spline_in_archive,
                        scale_spline_from_archive = options.scale_spline_from_archive,
                        plot = options.plot,
                        search_scan_range = options.search_scan_range,
                        output_plots_path = options.output_plots_path,
                        fit_type = options.spline_fit_type
                        )                  
    spl_fit()
    #
    #setattr(pyclass.gdict.r.user.sofia,"processing_steps",spl_fit.best_fit.name)
    # overplot on gildas
    setattr(pyclass.gdict.kosma, "spline_fit", spec.best_fit_spline) 
    setattr(pyclass.gdict.kosma, "spline_corrected", spec.corrected_intensity)    
    if options.plot=="gildas_plot":                 
        plot_fitted_spline_in_gildas()



if __name__ == "__main__":
    main()
