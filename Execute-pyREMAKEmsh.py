import pyREMAKEmsh
import json

# Auxiliary function for loading input data
def SaveJsonToDict(json_file):  
    
    with open(json_file) as f:
        input_dictionary = json.load(f)

    return input_dictionary

tolerance = 1e-4
input_data = 'InputData/GeometryData1.json'
input_dictionary = SaveJsonToDict(input_data)
pyREMAKEmsh.pyREMAKEmsh(input_dictionary, tolerance)