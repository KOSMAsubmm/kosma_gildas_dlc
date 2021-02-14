import pyclass
from sicparse import OptionParser
from collections import namedtuple
import pandas as pd
import os
from jinja2 import Template

from kosma_py_lib.pandas_index import *

def main():
    group_by_choices = ["SOURCE", "LINE", "TELESCOPE", "OFF1", "OFF2", "ENTRY", "NUMBER",
                        "BLOCK", "VERSION", "KIND", "QUALITY", "SCAN", "SUBSCAN"]
    param_requiring_single = ["source","telescope","line"]
    description='''
    script to sync the index numbers in an input file to a primar
    example command:
    merge_files /primary_in_file primary_file_name /in_file in_file_1
    '''
    parser = OptionParser(description=description)  
    parser.add_option("--primary_in_file", default = None,
                      dest="primary_in_file")
    parser.add_option("--file_to_merge_file", default = None,
                      dest="file_to_merge_file")
    parser.add_option("--reload", default = None, action="store_true",
                      dest="reload")                      
    if (not pyclass.gotgdict()):
      pyclass.get(verbose=False)
    try:
        global options
        (options, args) = parser.parse_args()
    except:
        pyclass.message(pyclass.seve.e, "merge_files", "Invalid option")
        pyclass.sicerror()
        return
    pyclass.comm("set var user")
    pyclass.comm("import sofia")
    # check are there index table for input files
    if options.primary_in_file is None:
        pyclass.message(pyclass.seve.e, "merge_files", "no primary file option given")
        return
    if options.file_to_merge_file is None:
        pyclass.message(pyclass.seve.e, "merge_files", "no input file option given")
        print description
        return        
    for filename in [options.primary_in_file,options.file_to_merge_file]:
        if not os.path.exists(filename):
            pyclass.message(pyclass.seve.e, "merge_files", "file {0} not found".format(filename))
            pyclass.sicerror()
            return
    pyclass.comm("file in \"{0}\"".format(options.primary_in_file))
    
    global df_primary
    global df_secondary
    if ("df_primary" not in globals()) or (options.reload):
        df_primary = get_index()
        latest_versions = df_primary.groupby(['number']).version.transform(max)
        df_primary = df_primary.loc[df_primary.version == latest_versions]
    # load in file index
    pyclass.comm("file in \"{0}\"".format(options.file_to_merge_file))
    #df_primary = get_index(observatory="SOFIA_GREAT")
    if ("df_secondary" not in globals()) or (options.reload):        
        df_secondary = get_index()
    # merge table to sort out
    df_merge = df_primary.merge(df_secondary,on=["scan","subscan","telescope","loff","boff"], suffixes = ("_primary","_secondary"))
    "get {0}"
    # write out in file with correct numbering
    template_string='''
file in "{{ input_filename }}"
file out "{{ tmp_output_filename }}" single /overwrite
find
{% for number,new_number in numbers %}
get {{number}} 
write {{new_number}} {% endfor %}

file in "{{ tmp_output_filename }}"
file out "{{ output_filename }}"
find
copy
exit
    '''
    t = Template(template_string)
    script = t.render(input_filename=options.file_to_merge_file, 
                      tmp_output_filename="tmp_merge_"+os.path.basename(options.file_to_merge_file), 
                      output_filename=options.primary_in_file,  
                      numbers=zip(df_merge.number_secondary.values,df_merge.number_primary.values))
    script_name = "./tmp_merge_renumber_output.class"
    with open(script_name, "w") as script_file:
        script_file.write(script)
    pyclass.message(pyclass.seve.i, "merge_files", "running class merge file")
    #class_cmd = ClassExecution()
    #class_cmd.run_class_script(script_name)
                   
                   
if __name__ == "__main__":
    main()
