# pyREMAKEmsh

## **Description**

**pyREMAKEmsh** is a meshing tool implemented in Python 3.7.6. adapted to the structures appearing in the ship structural analysis. The main contribution is the preprocessing algorithm which resolves the geometry of the object and divides it into plates where all of the intersections of the boundary edges are guaranteed to be the nodes of any mesh generated for such a geometry.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.

If you use **pyREMAKEmsh** in any program or publication, please acknowledge its authors by adding a reference to:

> Grubišić, Luka; Lacmanović, Domagoj; Prebeg, Pero; Tambača, Josip. Automatic mesh generation for structural analysis in naval architecture. Proceedings of the International Conference on Ships and Offshore Structures ICSOS 2021, 2021. [preprint](https://www.bib.irb.hr/1157014).

> Grubišić, Luka; Lacmanović, Domagoj; Prebeg, Pero; Tambača, Josip. pyREMAKEmsh – automatic mesh generation for ship structural analysis using open source software //Preprint. (scholarly, submitted, <a href="https://www.bib.irb.hr/1150462" title="pyREMAKEmsh">further bibliographic info</a>) (2021)

## **Dependencies**
* installed [Gmsh](https://gmsh.info/) (version 4.7.1 or above)
* installed [pygmsh](https://github.com/nschloe/pygmsh) (version 7.1.8 or above)

## **Documents**
- **input(json)** - dictionary containing the description of the geometry
- **output** - figures with mesh and time statistics, geo and msh files to be displayed by *Gmsh*

## **Example of usage**
```console
python3 Execute-pyREMAKEmsh.py ./InputData/GeometryData1.json
```

```python
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
```

![Geometry](/Figures/Geometry1.png "Geometry")
![Mesh](/Figures/Mesh1.png "Mesh")


## **Authors**
<span style="color:red"> **Lacmanović, Domagoj; Tambača, Josip and Grubišić, Luka** </span>

## **Licence**
This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
