import numpy as np
import matplotlib.pyplot as plt
#import pyclass
from sicparse import OptionParser
import sys
import pyclass
# Number of samplepoints


def main():
	parser = OptionParser()
	parser.add_option("--output_filename", dest="output_filename", nargs=1, default=False, type=str)
	parser.add_option("--min_fft_period", dest="min_fft_period", nargs=1, default=0, type=float)
	parser.add_option("--max_fft_period", dest="max_fft_period", nargs=1, default=400, type=float)
	parser.add_option("--no_plot", dest="no_plot", action="store_true", default=False)
	parser.add_option("--ignore_x_limits", dest="ignore_x_limits", action="store_true", default=False)
	parser.add_option("--fft_frequency_focus", dest="fft_frequency_focus", action="append", default=[])
	
	try:
		(options, args) = parser.parse_args()
	except:
		pyclass.message(pyclass.seve.e, "FFT", "Invalid option")
		pyclass.sicerror()
		return
	# sample spacing
	#
	if (not pyclass.gotgdict()):
		pyclass.get(verbose=False)
	y = pyclass.gdict.ry.__sicdata__
	rx = pyclass.gdict.rx.__sicdata__
	freq_res =  pyclass.gdict.r.head.spe.fres.__sicdata__
	vel_res =  pyclass.gdict.r.head.spe.vres.__sicdata__
	ref_channel =  pyclass.gdict.r.head.spe.rchan.__sicdata__
	channels = pyclass.gdict.channels.__sicdata__
	xmin = pyclass.gdict.user_xmin.__sicdata__
	xmax = pyclass.gdict.user_xmax.__sicdata__
	lower_axis_unit = pyclass.gdict.set.las.unit.__sicdata__[0]
	x_if = np.arange(len(y))*freq_res
	# apply user xmax and xmin		
	
	if lower_axis_unit == "F":
		x_orig = pyclass.gdict.r.head.spe.restf.__sicdata__ + (np.arange(len(y))-ref_channel)*freq_res
	elif lower_axis_unit == "I":
		x_orig = pyclass.gdict.r.head.spe.image.__sicdata__ + (np.arange(len(y))-ref_channel)*freq_res
	elif lower_axis_unit == "V":
		#x_orig = pyclass.gdict.r.head.spe.voff.__sicdata__ + (np.arange(len(y))-ref_channel)*vel_res
		x_orig = rx
	elif lower_axis_unit == "C":
		x_orig = np.arange(len(y))
	# detect total axis plot
	if (xmax== 1.0) & (xmin==0.0):
		index = np.arange(channels)
	else:
		index = np.where( (rx > min((xmin,xmax))) & (rx < max((xmin,xmax))))
	if options.ignore_x_limits:
	    index = np.arange(len(y))
	y = y[index]
	x_if = x_if[index]
	x_orig = x_orig[index]
	# crop out flagged data for fft
	index  = np.where(y!=-12345600000.0)
	y = y[index]
	x_orig = x_orig[index]
	x = x_if[index]
	# get x sample size
	N = len(x)/2
	# caclulate fft
	Yf    = abs(np.fft.fft(y))
	# convert fft frequenc
	freq = 1/np.fft.fftfreq(len(Yf), abs(freq_res))	
    # only show half of FFT
	Yf   = Yf[np.arange(len(Yf)/2)]
	freq = freq[np.arange(len(Yf)/2)]
	# show a plot of the data and the FFT
	# return power at selected frequency
	if len(options.fft_frequency_focus) > 0:
	    fft_channel_focus = []
	    max_freq = np.max(freq[np.isfinite(freq)])
	    min_freq = np.min(freq[np.isfinite(freq)])
	    for freq_focus in options.fft_frequency_focus:
	        # get power of nearest channel
	        freq_focus = float(freq_focus)
	        if float(freq_focus) > max_freq:
	            pyclass.message(pyclass.seve.e, "FFT", "{0} beyond max of FFT range".format(freq_focus))
	            continue
	        if float(freq_focus) < min_freq:
	            pyclass.message(pyclass.seve.e, "FFT", "{0} beyond min of FFT range".format(freq_focus))
	            continue
	        nearest_index = np.argmin(np.abs(freq-freq_focus))
	        nearest_freq = freq[nearest_index]
	        nearest_fft_power = Yf[nearest_index]
	        fft_channel_focus.append([freq_focus,nearest_freq,nearest_fft_power])
	    #
	    class_param = "fft_focus"
	    try:
	        pyclass.comm("del /var {0}".format(class_param))
	    except:
	        pass
	    fft_channel_focus = np.array(fft_channel_focus)
	    pyclass.comm("def real {0}[{1},{2}] /global".format(class_param,fft_channel_focus.shape[1],fft_channel_focus.shape[0]))
	    setattr(pyclass.gdict, class_param, fft_channel_focus)
	if options.no_plot:
	    return
	x_label_dict = {}
	x_label_dict["V"] = "Velocity (km/s)"
	x_label_dict["C"] = "Channels"
	x_label_dict["F"] = "Rest Frequency (MHz)"
	lower_axis_unit = pyclass.gdict.set.las.unit.__sicdata__[0]
	lower_axis_label = x_label_dict.get(lower_axis_unit,lower_axis_unit)
	fig, (ax_spectra,ax_fft) = plt.subplots(2,1)
	#
	fig.suptitle("{0.source}, line: {0.line}, {0.telescope} \n {0.scan}:{0.subscan}".format(pyclass.gdict))
	#
	ax_spectra.plot(x_orig,y)
	ax_spectra.set_ylabel("Antenna Temperature (K)")
	ax_spectra.set_xlabel("{0}".format(lower_axis_label))
	# crop out 
	# print options.min_fft_period,options.max_fft_period		
	index = np.where((freq>=options.min_fft_period) & (freq<=options.max_fft_period))
	ax_fft.semilogx(freq[index],abs(Yf[index]),drawstyle="steps-mid")
	ax_fft.set_ylabel("FFT power (a.u.)")
	ax_fft.set_xlabel("StWv Period (MHz)")
	# overplot selected bins
	if len(options.fft_frequency_focus) > 0:
	    for freq_focus in fft_channel_focus[:,0]:
	        ax_fft.axvline(x=float(freq_focus),color='r')
	    ax_fft.plot(fft_channel_focus[:,2], fft_channel_focus[:,3],'ro')
	#ax_fft.loglog(freq,abs(Yf))
    #ax_fft.set_xlim((fft_min,fft_max))
	if options.output_filename is False:
		plt.show()
	elif options.output_filename.lower()=="true":
		output_filename="fft_{0.source}_{0.line}_{0.telescope}_{0.scan}_{0.subscan}.png".format(pyclass.gdict).replace(" ","")
		print "saving to {0}".format(output_filename)		
		plt.savefig(output_filename)
	else:
		print "saving to {0}".format(options.output_filename)
   		plt.savefig(options.output_filename)
   


if __name__ == "__main__":
    main()
