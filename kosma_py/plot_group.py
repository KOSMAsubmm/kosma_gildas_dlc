import pyclass
from sicparse import OptionParser
from collections import namedtuple

def main():
    group_by_choices = ["SOURCE", "LINE", "TELESCOPE", "OFF1", "OFF2", "ENTRY", "NUMBER",
                        "BLOCK", "VERSION", "KIND", "QUALITY", "SCAN", "SUBSCAN"]
    param_requiring_single = ["source","telescope","line"]
    description='''
    # average spectra telescope,source and subscan, base 0 to bring to zero and smooth result by 20 bins
    plot_group /group_by telescope /group_by source /smooth_box 20 /base 0 /group_by subscan

    # show index map plot for each telescope and source grouping
    # careful with overlapping groups
    plot_group /group_by telescope /group_by source /plot_type index_map
    '''
    parser = OptionParser(description=description)
    parser.add_option("-g", "--group_by", default = [],
                      dest="group_by", action="append")
    parser.add_option("--divide_groups", default = 1, type = int,
                      dest="divide_groups")
    parser.add_option("-b", "--base", default=False,
                      dest="base", action="store_true")
    parser.add_option("-s", "--smooth_box", default=0,
                      dest="smooth_box")
    parser.add_option("-w", "--write_out_spectra", default=False,
                      dest="write_out_spectra", action="store_true")
    parser.add_option("--save_plot_per_group", default=False,
                      dest="save_plot_per_group", action="store_true")
    parser.add_option("--output_folder", default="./",
                      dest="output_folder", type=str)
    parser.add_option("--tag", default=[],
                      dest="tag", action="append")
    parser.add_option("-p", "--plot_type", default="average",
                      dest="plot_type", choices=["average","index_map","stamp"])
    if (not pyclass.gotgdict()):
      pyclass.get(verbose=False)
    try:
        global options
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "filter", "Invalid option")
        pyclass.sicerror()
        return
    pyclass.comm("set var user")
    pyclass.comm("import sofia")
    pyclass.comm("set align c")
    found_values = []
    # determine parameter type
    # check are input parameters in kosma structure
    # if not run kosma\get_meta function with input param
    #
    pyclass.comm("list /toc {0}".format(" ".join(options.group_by)))

    pyclass.comm("clear".format(options.group_by))
    #
    telescopes=[]
    pen_dict = {}
    legend_dict = {}
    legend_offset = {}
    arrays=[]
    columns, rows = pyclass.gdict.toc.setup.shape
    for i in range(rows):
        row_values = dict(zip(options.group_by,[ele.strip() for ele in pyclass.gdict.toc.setup[:,i]]))
        for group,divide_group in enumerate(range(options.divide_groups)):
            pen_color = i+group
            find_command = "find"
            for param,value in row_values.items():
                if " " in value:
                  value = value.replace(" ","\ ")
                if param.lower() in param_requiring_single:
                    find_command+=" /{0} {1}".format(param, value)
                else:
                    find_command+=" /{0} {1} {1}".format(param, value)
            print find_command
            pyclass.comm(find_command)
            if pyclass.gdict.found == 0:
                pyclass.message(pyclass.seve.e, "filter", "nothing found for {0}".format(find_command))
                continue
            fraction_of_found = "{0}/{1}".format(divide_group,options.divide_groups)
            if options.divide_groups>1:
                start_number = int(pyclass.gdict.found*float(divide_group)/options.divide_groups)
                end_number = int(pyclass.gdict.found*float(divide_group+1)/options.divide_groups)
                find_command+=" /number {0} {1}".format(pyclass.gdict.idx.num.__sicdata__[start_number],
                                                        pyclass.gdict.idx.num.__sicdata__[end_number-1]
                                                        )
                pyclass.comm(find_command)
            if options.plot_type=="average":
                pyclass.comm("average /nocheck")
                if options.smooth_box > 1:
                    pyclass.comm("smooth box {0}".format(options.smooth_box))
                if options.base:
                    try:
                        pyclass.comm("base 0")
                    except:
                        print "failed to fit"
                pen_key = "{0}".format(" ".join([row_values[opt] for opt in options.group_by if "telescope"!=opt]))
                if "telescope" in options.group_by:
                    telescope = row_values["telescope"]
                    try:
                        array = telescope.split("_")[0]
                    except:
                        array = ""   
                    #pen = pen_dict.setdefault(array,i%6)
                    if fraction_of_found !="0/1":
                        pen_key += " {0}".format(fraction_of_found)
                    elif len(options.group_by)==1:
                        pen_key += " {0}".format(array)
                    if len(pen_dict.values()) == 0:
                        pen_color=0
                    else:
                        pen_color=(max(pen_dict.values())+1)%6
                    #
                    pen = pen_dict.setdefault(pen_key,pen_color)
                    pyclass.comm("@plot_hexagon_pixel {0} {1} spectrum".format(row_values["telescope"][-3],pen))
                    #
                    legend = " ".join([row_values[opt] for opt in options.group_by])
                    #legend_offset[pen_key] =
                else:
                    pen = pen_color%6
                    if fraction_of_found !="0/1":
                        pen_key += " {0}".format(fraction_of_found)
                    pen = pen_dict.setdefault(pen_key,pen_color%6)
                    pyclass.comm("pen {0}".format(pen))
                    if pen_color==0:
                        pyclass.comm("plot")
                    else:
                        pyclass.comm("spec")
                #
                # write out averaged spectra
                if options.write_out_spectra:
                    pyclass.comm("write")
            elif options.plot_type=="index_map":
                #
                pyclass.comm("@plot_hexagon_pixel {0} {1} {2}".format(row_values["telescope"][-3],0,"index"))
            elif options.plot_type=="stamp":
                pyclass.comm("@plot_hexagon_pixel {0} {1} {2}".format(row_values["telescope"][-3],0,"stamp"))
            #

        if options.save_plot_per_group:
            save_plot=True
            if "telescope" in options.group_by:
                if row_values["telescope"] not in telescopes:
                    telescopes.append(row_values["telescope"])
                    save_plot=False
                # check all telescopes processes
                print telescopes,save_plot,pyclass.gdict.toc.tele                                
                if len(set(pyclass.gdict.toc.tele)-set(telescopes))==0:
                    save_plot=True 
                    telescopes=[]
                print telescopes,save_plot,pyclass.gdict.toc.tele                    
            #
            if save_plot:
                tag="_".join(options.tag)
                tag+="_"
                tag+="_".join(["{0}_{1}".format(key,value) for key,value in row_values.items() if key != "telescope"])
                #
                group_plot_filename = "{0}/group_plot_{1}_{2}".format(options.output_folder,options.plot_type,tag)
                pyclass.comm("ha \"{0}\" /device png /overwrite".format(group_plot_filename))
                pyclass.comm('draw text 0 {0} "{1}"'.format(20,"_".join(row_values.values())))
                pyclass.comm("clear")
                #            
    if not options.save_plot_per_group:            
      for i, (key, value) in enumerate(pen_dict.items()):
          pyclass.comm("pen {0}".format(value))
          pyclass.comm('draw text 0 {0} "{1}"'.format(20-i,key))
    #
    pyclass.comm("find")





if __name__ == "__main__":
    main()
