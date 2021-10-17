##########################
# Execute-pyREMAKEmsh.py #
##########################

import pyREMAKEmsh
import sys
import json

# Auxiliary function for loading input data
def SaveJsonToDict(json_file):  
    
    with open(json_file) as f:
        input_dictionary = json.load(f)

    return input_dictionary

tolerance = 1e-4
input_data_name = str(sys.argv[1])
input_dictionary = SaveJsonToDict(input_data_name)

# Remove everyting from input string except name of the file
input_data_name = input_data_name.split("/")
input_data_name = input_data_name[-1][0:-5]
pyREMAKEmsh.pyREMAKEmsh(input_dictionary, tolerance, input_data_name)