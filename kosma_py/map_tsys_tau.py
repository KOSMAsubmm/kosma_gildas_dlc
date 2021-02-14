from __future__ import print_function
import pyclass
import pgutils
from kosma_py_lib.pandas_index import get_index
from sicparse import OptionParser

def main():
    parser = OptionParser()
    try:
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "PCA", "Invalid option")
        pyclass.sicerror()
        return
    if (not pyclass.gotgdict()):
        pyclass.get(verbose=False)

    NON_SCIENCE_DATA = ["SKY-DIFF", "TAU_SIG", "TAU_IMG", "TAU_AVG",
                        "S-H_OBS", "S-H_FIT", "TSYS", "CAL_SIG", "CAL_IMG", "TREC (SSB)"]

    df = get_index()
    sources = df.source.unique()
    science_sources = [source for source in sources if source not in NON_SCIENCE_DATA]
    groups = df.groupby(["scan", "subscan", "telescope"])
    for name, group in groups:
        scan = name[0]
        subscan = name[1]
        telescope = name[2]
        group_sources = group.source.unique()
        for source in science_sources:
            if source not in group_sources:
                continue
            print(scan,subscan,telescope,  group.source.unique())
            TSYS_spectra = df.loc[(df.scan == scan) &
                                  (df.telescope == telescope) &
                                  (df.subscan == subscan-1) &
                                  (df["source"]=="TSYS")].index
            TAU_spectra = df.loc[(df.scan == scan) &
                                 (df.telescope == telescope) &
                                 (df.subscan == subscan-1) &
                                 (df["source"]=="TAU_SIG")].index
            if len(TSYS_spectra != 0) and len(TAU_spectra != 0):

                source_index = group.loc[group["source"]==str(source)].index
                pyclass.comm("get {}".format(TSYS_spectra[0]))
                try:
                    pyclass.comm("del /var tsys_spec")
                except pgutils.PygildasError:
                    pass
                try:
                    pyclass.comm("del /var tsys_bad")
                except pgutils.PygildasError:
                    pass
                pyclass.comm("def real tsys_bad")
                pyclass.comm("let tsys_bad r%head%spe%bad")

                pyclass.comm("def real tsys_spec /like ry")
                pyclass.comm("let tsys_spec ry")
                pyclass.comm("get {}".format(TAU_spectra[0]))
                try:
                    pyclass.comm("del /var tau_bad")
                except pgutils.PygildasError:
                    pass
                try:
                    pyclass.comm("del /var tau_spec")
                except pgutils.PygildasError:
                    pass
                pyclass.comm("def real tau_bad")
                pyclass.comm("let tau_bad r%head%spe%bad")

                pyclass.comm("def real tau_spec /like ry")
                pyclass.comm("let tau_spec ry")
                for number in source_index:
                    print(number)
                    # get the tsys spectrum
                    pyclass.comm("get {}".format(number))
                    pyclass.comm("associate tsys tsys_spec /bad tsys_bad")
                    pyclass.comm("associate tau tau_spec /bad tau_bad")
                    pyclass.comm("write")
        for source in NON_SCIENCE_DATA:
            source_index = group.loc[group["source"]==str(source)].index
            if len(source_index) > 1:
                for number in source_index:
                    pyclass.comm("get {}".format(number))
                    pyclass.comm("write")

main()
