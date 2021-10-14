# pyREMAKEmsh

## **Description**

**pyREMAKEmsh** is a meshing tool implemented in Python 3.7.6. adapted to the structures appearing in the ship structural analysis. The main contribution is the preprocessing algorithm which resolves the geometry of the object and divides it into plates where all of the intersections of the boundary edges are guaranteed to be the nodes of any mesh generated for such a geometry.

This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html) for more details.

If you use **pyREMAKEmsh** in any program or publication, please acknowledge its authors by adding a reference to:

> Grubišić, Luka; Lacmanović, Domagoj; Prebeg, Pero; Tambača, Josip. pyREMAKEmsh – automatic mesh generation for ship structural analysis using open source software //Preprint (2021) (scholarly, submitted) <a href="https://www.bib.irb.hr/1150462" title="pyREMAKEmsh">bibliographic info</a>

## **Dependencies**
* installed [Gmsh](https://gmsh.info/) (version 4.7.1 or above)
* installed [pygmsh](https://github.com/nschloe/pygmsh) (version 7.1.8 or above)

## **Documents**
- **input(json)** - dictionary containing the description of the geometry
- **output** - geo and msh files to be displayed by *Gmsh*

## **Example of usage**
```python
import pyREMAKEmsh
import json

# Auxiliary function for loading input data
def SaveJsonToDict(json_file):  
    
    with open(json_file) as f:
        input_dictionary = json.load(f)

    return input_dictionary

tolerance = 1e-4
input_data = 'GeometryData1.json'
input_dictionary = SaveJsonToDict(input_data)
pyREMAKEmsh.pyREMAKEmsh(input_dictionary, tolerance)

```

![Geometry](/Figures/Geometry1.png "Geometry")
![Mesh](/Figures/Mesh1.png "Mesh")


## **Authors**
<span style="color:red"> **Lacmanović, Domagoj; Tambača, Josip and Grubišić, Luka** </span>

## **Licence**
This software is published under the [GPLv3 license](https://www.gnu.org/licenses/gpl-3.0.en.html).
