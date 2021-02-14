
import pyclass
from sicparse import OptionParser
import numpy as np
import matplotlib.pyplot as pyplt
from matplotlib.collections import PathCollection
#import get_unique_header_values
#from matplotlib.pyplot import figure,show

def determine_meta_type(value):
    if "S" in value.dtype.__str__():
        return str
    elif "f" in value.dtype.__str__():
        return float
    elif "i" in value.dtype.__str__():
        return int
     
def get_meta_value(meta_entry):
    try:
        meta_value = eval("pyclass.gdict.{0}".format(meta_entry.lower().replace("%","."))).__sicdata__
        return meta_value
    except:
        pyclass.message(pyclass.seve.e, 
                        "filter",
                        "pyclass.gdict.{0} not found in gdict".format(meta_entry.lower().replace("%",".")))
        return False
    
def onpick(event):
   ind = event.ind
   #print ind, dir(event),isinstance(event.artist,PathCollection)
   #print event.artist.__dict__
   index = map(int, event.artist._offsets[:,0][ind])
   print index
   print 'onpick scatter:', index, np.take(pyclass.gdict.kosma.scan_unique, index), np.take(pyclass.gdict.kosma.subscan_unique,index), np.take(event.artist._offsets[:,0], ind)
   scan_numbers = np.take(pyclass.gdict.kosma.scan_unique, index)
   subscan_numbers = np.take(pyclass.gdict.kosma.subscan_unique, index)  
   numbers = np.take(pyclass.gdict.kosma.number_unique, index)
   if options.linked_plot=="S-H":
      pyclass.comm('set scan {0} {0}'.format(scan_numbers[0]))
      pyclass.comm('set subscan {0} {0}'.format(subscan_numbers[0]))
      pyclass.comm('set source S-H*')
      pyclass.comm('find')
      pyclass.comm('pen 0')      
      if pyclass.gdict.found==0:
         pyclass.comm('cl')
         pyclass.comm('draw text 14 14 "no S-H found for {0}:{1}"'.format(scan_numbers[0],subscan_numbers[0]-1))
         return   
      pyclass.comm('@plot_hexagon {0}'.format("LFAV"))      
      # TODO how do i reset the folder
      pyclass.comm('set scan')
      pyclass.comm('set subscan')
      pyclass.comm('set source')
      pyclass.comm('find')          
   
   elif options.linked_plot=="S-Hsingle":
      pyclass.comm('set scan {0} {0}'.format(scan_numbers[0]))
      pyclass.comm('set subscan {0} {0}'.format(subscan_numbers[0]))
      pyclass.comm('set source S-H*')
      pyclass.comm('find')
      pyclass.comm('pen 0')      
      if pyclass.gdict.found==0:
         pyclass.comm('cl')
         pyclass.comm('draw text 14 14 "no S-H found for {0}:{1}"'.format(scan_numbers[0],subscan_numbers[0]-1))
         return
      pyclass.comm('get f')
      # set mode externally
      pyclass.comm('pl')
      pyclass.comm('get n')
      pyclass.comm('pen 1')
      pyclass.comm('spec')
      # TODO how do i reset the folder
      pyclass.comm('set scan')
      pyclass.comm('set subscan')
      pyclass.comm('set source')
      pyclass.comm('find')          
   elif  options.linked_plot=="index_plot":
      pyclass.comm('set scan {0} {0}'.format(scan_numbers[0]))
      pyclass.comm('set subscan {0} {0}'.format(subscan_numbers[0]))
      pyclass.comm('find')
      pyclass.comm('pen 0')      
      if pyclass.gdict.found==0:
         pyclass.comm('cl')
         pyclass.comm('draw text 14 14 "nothing found for {0}:{1}"'.format(scan_numbers[0],subscan_numbers[0]-1))         
         return         
      pyclass.comm('load /nocheck')
      pyclass.comm('plot /index')
      return
   elif   options.linked_plot=="single":
      pyclass.comm('get {0}'.format(numbers[0]))
      pyclass.comm('pl')
      return

def main():
    parser = OptionParser()
    parser.add_option("-m", "--meta_entry", dest="meta_entry", 
                      nargs=1, default=None
                      )    
    parser.add_option("-f", "--filter_per", default = "scan",
                      dest="filter_per", nargs=1, choices = ["scan","subscan","number"])
    parser.add_option("-g", "--group_by", default = None,
                      dest="group_by", nargs=1)
    parser.add_option("-l", "--linked_plot", default = "S-H",
                      dest="linked_plot", nargs=1, choices = ["S-H","index_plot", "single"])
    parser.add_option("-r", "--rescan", default=False,
                      dest="rescan", action="store_true")  
    if (not pyclass.gotgdict()):
      pyclass.get(verbose=False)
    try:
        global options
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "filter", "Invalid option")
        pyclass.sicerror()
        return
    if options.meta_entry is None:
        print "no options given, see /help"
        return 
    pyclass.comm("set var user")
    pyclass.comm("import sofia")
    found_values = []
    # determine parameter type
    # check are input parameters in kosma structure
    # if not run kosma\get_meta function with input param
    meta_type_dict = {}
    present = []
    unique_entries_needed = [options.meta_entry,"scan","subscan","number"]
    if options.group_by is not None:
       unique_entries_needed.append(options.group_by)
    for unique_entry in unique_entries_needed:
        kosma_structure_name = unique_entry.split("%")[-1].lower()+"_unique"
        if  getattr(pyclass.gdict.kosma, kosma_structure_name, None) is None:
            present.append(False)
        else:
            present.append(True)
    # running get meta function
    present.append(not options.rescan)
    if not all(present):
       meta_string = " ".join(["/meta_entry "+entry for entry in unique_entries_needed])
       get_meta_cmd="get_meta {0} /filter_per {1}".format(meta_string, options.filter_per)
       import time
       pyclass.comm(get_meta_cmd)    
    # setup plot
    #
    #if options.filter_per == "number":
    #    options.linked_plot = "number"
    print "opening figure"
    fig = pyplt.figure()
    print "opening figure"    
    ax_data = fig.add_subplot(111)
    data = getattr(pyclass.gdict.kosma, options.meta_entry.split("%")[-1].lower()+"_unique")        
    if options.group_by is None:
       ax_data.scatter(np.arange(len(data)), data, picker=True)
    else:       
       group_by_name = options.group_by.split("%")[-1].lower()+"_unique"
       group_by_data = getattr(pyclass.gdict.kosma, group_by_name)
       for group_by_value in list(set(group_by_data)):
           index, = np.where(group_by_data==group_by_value)
           ax_data.scatter(np.arange(len(data))[index], data[index], label=str(group_by_value), picker=True)
       #
       pyplt.legend(loc=8,fontsize="small")         
    ax_data.set_ylabel(options.meta_entry)

    fig.canvas.mpl_connect('pick_event', onpick)
    pyplt.show()
    pyclass.comm("find")
    



if __name__ == "__main__":
    main()
