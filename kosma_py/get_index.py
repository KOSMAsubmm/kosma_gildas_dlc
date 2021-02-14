from __future__ import print_function
import pyclass
from sicparse import OptionParser
from kosma_py_lib.pandas_index import get_index, get_class_windows


def main():
    parser = OptionParser()
    parser.add_option("-f", "--reload", dest="reload",
                      default=False, action='store_true'
                      )
    parser.add_option("-i", "--input_file", dest="input_file",
                      default=None)
    parser.add_option("-p", "--parameter", dest="additional_meta_parameters",
                      default=None, action='append'
                      )
    parser.add_option("-r", "--get_rms", dest="get_rms",
                      default=False, action='store_true'
                      )
    parser.add_option("-s", "--get_stats", dest="get_stats",
                      default=False, action='store_true'
                      )
    parser.add_option("-o", "--output_file", dest="output_file",
                      nargs=1, default=None)
    parser.add_option("-l", "--get_stats_over_range",
                      dest="get_stats_over_range",
                      nargs=3, default=[], action="append",
                      )
    parser.add_option("-t", "--generate_time_column",
                      dest="generate_time_column",
                      default=False, action='store_true'
                      )
    parser.add_option("--fft_focus_freq", dest="fft_focus_freq",
                      default=[], action='append',
                      help="gather power at this FFT frequency"
                      )
    # parser.add_option("-f", "--filter_per", default = "scan",
    #                  dest="filter_per", nargs=1,
    #                  choices = ["scan","subscan", "number"])
    if (not pyclass.gotgdict()):
        pyclass.get(verbose=False)
    try:
        (options, args) = parser.parse_args()
    except KeyError:
        pyclass.message(pyclass.seve.e, "filter", "Invalid option")
        pyclass.sicerror()
        return
    windows = get_class_windows()
    #
    if len(windows) == 0:
        if (options.get_rms is True) or (options.get_stats is not False):
            print("no windows defined and rms or stats option with "
                  "baselining selected, exiting")
            print("define windows and retry")
            return
    #
    if (options.additional_meta_parameters is None) and (options.generate_time_column is True):
        options.additional_meta_parameters = ["R%HEAD%GEN%CDOBS", "utobs"]
    elif (options.additional_meta_parameters is not None) and (options.generate_time_column is True):
        options.additional_meta_parameters.extend(["R%HEAD%GEN%CDOBS", "utobs"])
    #
    global df
    df = get_index(
        observatory="SOFIA_GREAT",
        additional_meta_parameters=options.additional_meta_parameters,
        force_reload=options.reload,
        get_rms=options.get_rms,
        get_stats=options.get_stats,
        get_fft_power=options.fft_focus_freq,
        get_stats_over_range=options.get_stats_over_range,
        input_file=options.input_file,
        output_file=options.output_file
    )

    if df is None:
        return
    if str(pyclass.gdict.set.las.source.__sicdata__).strip() != "*":
        source = "{}".format(
            str(pyclass.gdict.set.las.source.__sicdata__).strip()
            )
        df = df[df["source"] == source]
    print(df.columns)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        pyclass.message(pyclass.seve.e, "ERROR", e.message)
