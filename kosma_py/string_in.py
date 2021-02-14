import glob

import pyclass
from sicparse import OptionParser
import sys

def main():
    if len(sys.argv)!=3:
        print "not enough arguments"
        return
    #
    if (not pyclass.gotgdict()):
      pyclass.get(verbose=False)
    #
    sys.argv = [arg.replace("\"","") for arg in sys.argv]
    string_found = sys.argv[1] in sys.argv[2]
    #
    Sic.comm("let kosma%is_string_found {0}".format(str(string_found)))
    if string_found:
        Sic.comm('let kosma%string_found {0}'.format(sys.argv[1]))
    else:
        Sic.comm('let kosma%string_found "{0}"'.format(" "))
    #
    return

if __name__ == "__main__":
    main()
