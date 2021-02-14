
import pyclass
from sicparse import OptionParser

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
        print "pyclass.gdict.{0} not found in gdict for first spectra, check name or first spectra in index".format(meta_entry.lower().replace("%","."))
        return False
    
    
def main():
    parser = OptionParser()
    parser.add_option("-m", "--meta_entry", dest="meta_entries", 
                      nargs=1, default=[], action='append'
                      )    
    parser.add_option("-f", "--filter_per", default = "scan",
                      dest="filter_per", nargs=1, choices = ["scan","subscan", "number"])
    parser.add_option("-w", "--write_results_to_file", default = None,
                      dest="write_results_to_file")
    if (not pyclass.gotgdict()):
      pyclass.get(verbose=False)
    try:
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "filter", "Invalid option")
        pyclass.sicerror()
        return
    if len(options.meta_entries)==0:
        print "no meta parameters given"
        return
    pyclass.comm("find")
    pyclass.comm("set var user")
    pyclass.comm("import sofia")    
    pyclass.comm("def structure kosma")
    found_values = []
    # determine parameter type
    pyclass.comm("get f")
    meta_type_dict = {}
    for meta_entry in options.meta_entries:
        meta_value = get_meta_value(meta_entry)
        if meta_value is False:
            return
        meta_type  = determine_meta_type(meta_value)
        meta_value = meta_type(meta_value)
        meta_type_dict[meta_entry] = meta_type
    #
    if options.filter_per == "scan":
        for scan in sorted(set(pyclass.gdict.idx.scan)):
            pyclass.comm("find /scan {0}".format(scan))
            pyclass.comm("get f")
            # check meta data
            meta_values = []
            for meta_entry in options.meta_entries:
                meta_value = meta_type_dict[meta_entry](get_meta_value(meta_entry))
                meta_values.append(meta_value)                
            meta_values = tuple(meta_values)            
            if meta_values not in found_values:
                found_values.append(meta_values)
    elif options.filter_per == "subscan":
        for scan,subscan in sorted(set(zip(pyclass.gdict.idx.scan,pyclass.gdict.idx.subscan))):
            pyclass.comm("find /scan {0} /subscan {1}".format(scan,subscan))
            pyclass.comm("get f")
            # check meta data
            meta_values = []
            for meta_entry in options.meta_entries:
                meta_value = meta_type_dict[meta_entry](get_meta_value(meta_entry))
                meta_values.append(meta_value)                
            meta_values = tuple(meta_values)
            if meta_values not in found_values:
                found_values.append(meta_values)
    elif options.filter_per == "number":
        for number in pyclass.gdict.idx.num.__sicdata__:
            pyclass.comm("get {0}".format(number))
            # check meta data
            meta_values = []
            for meta_entry in options.meta_entries:
                meta_value = meta_type_dict[meta_entry](get_meta_value(meta_entry))
                meta_values.append(meta_value)                
            meta_values = tuple(meta_values)
            if meta_values not in found_values:
                found_values.append(meta_values)
    # delete previous unique parameters
    #try:
    unique_params = ["kosma%"+param for param in dir(pyclass.gdict.kosma) if "_unique" in param]
    for param in unique_params:
       print "del /var "+param
       pyclass.comm("del /var "+param)
    if hasattr(pyclass.gdict.kosma, "kosma%unique_size"):
        pyclass.comm("def integer kosma%unique_size")
    setattr(pyclass.gdict.kosma, "unique_size", len(found_values))
    
    #except:
    #pass
    #
    if len(found_values)>0:
        for i,meta_entry in enumerate(options.meta_entries):
            class_param = "kosma%{0}_unique".format(meta_entry.split("%")[-1])
            try:
                pyclass.comm("del /var "+class_param)
            except:
                pass
            meta_type = meta_type_dict[meta_entry]
            print meta_type
            if meta_type==float:
                pyclass.comm("def double {0}[{1}]".format(class_param,len(found_values)))
                found_values_entry = [float(value[i]) for value in found_values]
            elif meta_type==int:
                pyclass.comm("def integer {0}[{1}]".format(class_param,len(found_values)))
                found_values_entry = [int(value[i]) for value in found_values]
            else:
                pyclass.comm("def char*100 {0}[{1}]".format(class_param,len(found_values)))#
                found_values_entry = [str(value[i]).strip() for value in found_values]
            #
            print "writing unique data to {0}".format(class_param.upper())
            setattr(pyclass.gdict.kosma, class_param.split("%")[-1].lower(), found_values_entry)
    #
    if options.write_results_to_file is not None:
        output_filename = "{0}".format(options.write_results_to_file)
        output_file = open(output_filename,'w')
        # print header
        print "writing results to {0}".format(output_filename)
        output_file.write(",".join(options.meta_entries)+"\n")
        for values in found_values:
            output_file.write(",".join([str(value).strip() for value in values])+"\n")
        output_file.close()
        
    return


if __name__ == "__main__":
    main()
