import pyclass
from sicparse import OptionParser
import re
import sys



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

def compare_values(meta_value_scan, meta_operator, meta_filter_value):
    #
    if meta_operator=="==":
        return meta_value_scan==meta_filter_value
    elif meta_operator==">=":
        return meta_value_scan>=meta_filter_value
    elif meta_operator=="<=":
        return meta_value_scan<=meta_filter_value
    elif meta_operator==">":
        return meta_value_scan>meta_filter_value
    elif meta_operator=="<":
        return meta_value_scan<meta_filter_value
    elif meta_operator=="!=":
        return meta_value_scan!=meta_filter_value
    elif meta_operator==" in ":
        return meta_filter_value in meta_value_scan

def filter():
    #
    possible_operators = ["==",">=","<=",">","<","\!=", " in "]
    regex_string = "|".join(possible_operators)
    #
    conditions = []
    for meta_filter in options.meta_filter:
        if re.match("(.+)({0})(.*)".format(regex_string),meta_filter) is None:
            pyclass.message(pyclass.seve.e,
                            "regex",
                            "operator not valid in filter {0}".format(meta_filter))
            return
        meta_entry, meta_operator, meta_filter_value = re.match("(.+)({0})(.*)".format(regex_string),meta_filter).groups()
        meta_value_scan = get_meta_value(meta_entry)
        if meta_value_scan is False:
            return
        # convert value to appropriate python equivalent
        data_type = determine_meta_type(meta_value_scan)
        meta_value_scan = data_type(meta_value_scan)
        meta_filter_value = data_type(meta_filter_value)
        # strip blank space from string
        if type(meta_value_scan) is str:
            meta_value_scan = meta_value_scan.strip()
            if meta_operator not in ["==","!=", ' in ']:
                pyclass.message(pyclass.seve.e,
                                    "regex",
                                    "{0}\n only == operator usable with strings".format(meta_filter))
                sys.exit()
        #
        conditions.append(compare_values(meta_value_scan, meta_operator,meta_filter_value ))
    if all(conditions):
        return True
    else:
        return False


def main():
    parser = OptionParser()
    parser.add_option("-m", "--meta_filter", dest="meta_filter",
                      nargs=1, default=[], action='append'
                      )
    parser.add_option("-f", "--filter_per", default='scan',
                      dest="filter_per", nargs=1, choices = ["scan","subscan"])
    #parser.add_option("-a", "--append_to_index", default=False,
    #                  dest="append_to_index", action="store_true")

    #
    if (not pyclass.gotgdict()):
      pyclass.get(verbose=False)
    global options
    try:
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "filter", "Invalid option")
        pyclass.sicerror()
        return
    pyclass.comm("set var user")
    pyclass.comm("import sofia")
    pyclass.comm("find")
    #
    found_scan_numbers = []
    if options.filter_per == "scan":
        for scan_number in list(set(pyclass.gdict.idx.scan)):
            pyclass.comm("find /scan {0}".format(scan_number))
            pyclass.comm("get f")
            if filter():
                if scan_number not in found_scan_numbers:
                    found_scan_numbers.append(scan_number)
        #
        if len(found_scan_numbers)>0:
            class_param = "kosma%meta_scans"
            try:
                pyclass.comm("del /var {0}".format(class_param))
            except:
                pass
            pyclass.comm("def integer {0}[{1}]".format(class_param,len(found_scan_numbers)))
            # check meta data
            setattr(pyclass.gdict.kosma, class_param.split("%")[-1], found_scan_numbers)
            # run find command
            for i, scan_number in enumerate(found_scan_numbers):
                #if (i==0) and (options.append_to_index is False):
                if (i==0):
                    pyclass.comm("find /scan {0}".format(scan_number))
                else:
                    pyclass.comm("find append /scan {0}".format(scan_number))
        else:
            # return nothing found
            pyclass.comm("find /scan 0")

    elif options.filter_per == "subscan":
        for scan,subscan in sorted(set(zip(pyclass.gdict.idx.scan,pyclass.gdict.idx.subscan))):
            pyclass.comm("find /scan {0} /subscan {1}".format(scan,subscan))
            pyclass.comm("get f")
            #
            if filter():
                found_scan_numbers.append((scan,subscan))
        # run find command
        if len(found_scan_numbers)>0:
            for i, (scan_number, subscan_number) in enumerate(found_scan_numbers):
                #if (i==0) and (options.append_to_index is False):
                if (i==0):
                    pyclass.comm("find /scan {0} /subscan {1}".format(scan_number,subscan_number))
                else:
                    pyclass.comm("find append /scan {0} /subscan {1}".format(scan_number,subscan_number))
        else:
            # return nothing found
            pyclass.comm("find /scan 0")

    #try:
    #    pyclass.comm("del /var kosma%file_list")
    #except:
    #    pass
    #pyclass.comm("def char*100 kosma%file_list[{}]".format(len(list_)))
    #pyclass.gdict.kosma.file_list = list_

if __name__ == "__main__":
    main()
