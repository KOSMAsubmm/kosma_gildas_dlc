
import matplotlib
#matplotlib.use('TkAgg')
import pyclass
from sicparse import OptionParser
import numpy as np
import matplotlib.pyplot as pyplt
from matplotlib.collections import PathCollection
from kosma_py_lib.pandas_index import *
from datetime import datetime
from scipy.optimize import curve_fit
import re
#from matplotlib import interactive
#interactive(True)

def onpick(event):
    ind = event.ind
    #print event
    #print len(event.artist.get_xdata()), len(event.artist.get_ydata())
    print event.artist._offsets[:,1]
    index = map(int, event.artist._offsets[:,0][ind])

    #return
    for i,(number,version) in enumerate(zip(filter_df.loc[index].index,filter_df.loc[index].version)):
        pyclass.comm('get {0} {1}'.format(number, version))
        #pyclass.comm('smooth box 20')
        if i==0:
            pyclass.comm('pen 0')
            pyclass.comm('clear')
            pyclass.comm('pl')
            pyclass.comm('draw window')
        else:
            pyclass.comm('pen {0}'.format(i%6))
            pyclass.comm('spec')
    return


# fitting function for gauss
def gauss(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def fit_gauss(x,y):
    max_idx = np.argmax(y)
    amp = y[max_idx]
    offset = x[max_idx]
    width = 1.0
    if max_idx==0:
        return
    #
    #popt, pcov = curve_fit(gauss,x,y, [amp, offset, width],bounds=[(0.01,10),(1,16),(0.01,10)])
    try:
        popt, pcov = curve_fit(gauss,x,y, [amp, offset, width],bounds=([0.01,1.0,0.01],[10.0,16.0,10.0]))
    except:
        return
    # fitted parameters
    fitted_amp = popt[0]
    fitted_offset = popt[1]
    fitted_width = popt[2]
    return {"amp":popt[0],"offset":popt[1],"width":popt[2]}


def main():
    parser = OptionParser()
    description='''
    script to generate ignore statements
    # you can chain table filters together
    # example plot rms, groupby telescope, generate histogram with 100 bins a limit of 0 to 10
    # and only plot spectra with version number 7 and spectra with an rms greater than 3 time the mean rms
    filter_index  /table_filters "version == 7" /table_filters "rms>rms.mean()"  /write_table
    filter_index  /table_filters "scan == 1213 and subscan > 2" /table_filters "rms>rms.mean()"  /write_table /array_name test_array_name
    filter_index  /table_filters  "telescope.str.contains('V_5')"
    '''
    parser = OptionParser(description)
    parser.add_option("-f", "--table_filters", default = [], action="append",
                      dest="table_filters", nargs=1,type=str)
    parser.add_option("--reload_python_var", default=False,
                      dest="reload_python_var", action="store_true")
    parser.add_option("-r", "--reload", default=False,
                      dest="reload", action="store_true")
    parser.add_option("-w", "--write_table", default=False,
                      dest="write_table", action="store_true")
    parser.add_option("-a", "--append_table", default=False,
                      dest="append_table", action="store_true")
    parser.add_option("-t", "--export_filename", default="filter_array.gildas",
                      dest="export_filename")
    parser.add_option("-l", "--array_name", default="",
                      dest="array_name")
    parser.add_option("-i", "--apply_find_index", default=False,
                     dest="apply_find_index", action="store_true")                      
    parser.add_option("-s", "--show_columns_available", default=False,
                     dest="show_columns_available", action="store_true")
    #
    if (not pyclass.gotgdict()):
      pyclass.get(verbose=False)
    try:
        global options
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "filter_index", "Invalid option")
        pyclass.sicerror()
        return
    #
    if (options.reload_python_var) or (options.reload):
        df = get_index(observatory="SOFIA_GREAT",
                       force_reload=options.reload)
        df = df.set_index("number")
    #
    pyclass.comm("set var user")
    pyclass.comm("import sofia")
    #
    #df = get_index(observatory="SOFIA_GREAT",
    #               additional_meta_parameters = options.additional_meta_parameters,
    #               force_reload=options.reload,get_rms=options.get_rms)
    global df
    if "df" not in globals():
        options.reload_python_var = True
    #else:
    #print "pandas table already in memory with following columns:"
    #print df.columns


    if (options.reload_python_var) or (options.reload):
        df = get_index(observatory="SOFIA_GREAT",
                       force_reload=options.reload)
        #df = df.set_index("number")
    #
    if df is None:
        return
    #
    if options.show_columns_available:
       print df.columns
       return 
    # filter df based on index numbers
    global filter_df
    if options.apply_find_index:
        filter_df = df.loc[list(pyclass.gdict.idx.num.__sicdata__)]
    else:
        filter_df = df
    # filter
    if len(options.table_filters)>0:
        # rms<10
        tables = []
        print options.table_filters
        for table_filter in options.table_filters:
            #try:
            if table_filter is None:
                pyclass.message(pyclass.seve.i, "filter_index", "too many arguments in filter, limited to 9 i think")
                continue
            rex = re.match("(.+):(.+)",table_filter)
            if rex is not None:
               table_filter,filter_description = rex.groups()            
            if True:
                pyclass.message(pyclass.seve.i, "filter_index", "{0}".format(table_filter,"\n".join(filter_df.columns)))
                if table_filter is None:
                    continue
                rows_found = filter_df[filter_df.eval(table_filter, engine="python")]
                for telescope,rows in rows_found.groupby(["telescope"]):                    
                    pyclass.message(pyclass.seve.i, "filter_index", "{0}: {1}:{2} found".format(table_filter, telescope,len(rows)))
                    pyclass.message(pyclass.seve.i, "filter_index", "{0}: {1}:{2}".format(table_filter, telescope,rows.scan.unique()))                    
                pyclass.message(pyclass.seve.i, "filter_index", "{0}: {1} found".format(table_filter,len(rows_found)))
                tables.append(rows_found)
            else:
            #except:
                pyclass.message(pyclass.seve.e, "filter_index", "{0} failed, check columns names:\n{1}".format(table_filter,"\n".join(filter_df.columns)))
                return
        #
        filter_df = pd.concat(tables,verify_integrity=False).drop_duplicates()
    #
    if len(filter_df)==0:
        pyclass.message(pyclass.seve.i, "filter_index", "no data found for filter settings".format())
        return
    #
    if (options.write_table) or (options.append_table):
        #
        if options.append_table:
            export_file = open(options.export_filename,'a')
        else:
            export_file = open(options.export_filename,'w')
        pyclass.message(pyclass.seve.i, "filter_index", "writing {0} spectra to {1} file".format(len(filter_df),options.export_filename))
        #message =  'ignoring {0} spectra\n'.format(len(filter_df))
        #ignore_file.write("say "+message)
        for index,row in filter_df.iterrows():
            export_file.write("{0}\n".format(index))
        export_file.close()
    else:
        pyclass.message(pyclass.seve.i, "filter_index", "{0} spectra in filter".format(len(filter_df),options.export_filename))
    # load
    if options.array_name!="":
        # load_file_into_array filename array_name
        old_data = []
        if hasattr(pyclass.gdict, options.array_name):
            old_data = list(getattr(pyclass.gdict, options.array_name))
            pyclass.comm("delete /var {0}".format(options.array_name))
            pyclass.comm("delete /var {0}_len".format(options.array_name))
        new_data = list(old_data)+list(filter_df.index.values)
        new_data = list(set(new_data))
        pyclass.comm("def int {0}[{1}] /global".format(options.array_name,len(new_data)))
        pyclass.comm("def int {0}_len /global".format(options.array_name,len(new_data)))
        setattr(pyclass.gdict, options.array_name,  new_data)
        setattr(pyclass.gdict, "{0}_len".format(options.array_name),  len(new_data))
        pyclass.message(pyclass.seve.i, "filter_index", "{0} spectra number written to {1}".format(len(filter_df),options.array_name.upper()))
        #pyclass.comm("@load_file_into_array {0} {1}".format(options.export_filename,options.array_name))#




if __name__ == "__main__":
    main()
