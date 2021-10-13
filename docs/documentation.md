# pyREMAKEmsh documentation

## **Description**

**pyREMAKEmsh** is a meshing tool implemented in Python 3.7.6. adapted to the structures appearing in the ship structural analysis. The main contribution is the preprocessing algorithm which resolves the geometry of the object and divides it into plates where all of the intersections of the boundary edges are guaranteed to be the nodes of any mesh generated for such a geometry. The result of this preprocessing algorithm is a discrete description of the geometry which can be meshed by the automatic frontal mesher according to the requirements of the classification society.  The resulting algorithm is implemented in Python, using open source [Gmsh](https://gmsh.info/) system together with the Open CASCADE module and [pygmsh](https://github.com/nschloe/pygmsh) which combines the power of Gmsh with the versatility of Python.

## **class pyREMAKEmsh**

## **<span style="font-size: 1.1rem">\_\_init\_\_</span>**
```python
def __init__(self, geometry_data, tol):
```
- **Description** : initializes object's state; creates dictionary with input data and sets tolerance for Gmsh

- **Input** :   
    - geometry_data (*dictionary*) : geometry data (points, stiffeneres, surfaces etc.)
    - tol (*int*) : tolerance for Gmsh
              
- **Output** : /

## **<span style="font-size: 1.1rem">HoleInfo</span>**
```python
def HoleInfo(self, surface_id):
```

- **Description** : returns hole information from input dictionary
- **Input** :
    -  surface_id (*int*) : unique id of surface with hole
- **Output** :
    - surface_holes_info (*list*) : list that contains information about hole (hole center, hole length in directions w.r.t. local coordinate system, hole radius and hole rotation)

## **<span style="font-size: 1.1rem">MakeHole</span>**
```python
def MakeHole(self,surface_id):
```
- **Description** : constructs hole as a new surface and updates geometry dictionary
- **Input** :
    -  surface_id (*int*) : unique id of surface with hole
- **Output** : updated dictionary containing geometry data

## **<span style="font-size: 1.1rem">ConstructGeometry</span>**
```python
def ConstructGeometry(self):
```
- **Description** : using pygmsh OpenCASCADE kernel constructs geometry, reclassifies geometry using boolean operations and constructs mesh
- **Input** : /
- **Output** : 
    - GeometryBeforeBooleanOperations.geo_unrolled (*geo*) : geometry before boolean operations; can be visualised in Gmsh
    - GeometryAfterBooleanOperations.geo_unrolled (*geo*) : geometry after boolean operations; can be visualised in Gmsh
    - NonRecombinedMesh.msh (*msh*) : triangle mesh of the geometry; can be visualised in Gmsh
    - RecombinedFinalMesh.msh (*msh*) : recomined mesh of the geometry; can be visualised in Gmsh

## **<span style="font-size: 1.1rem">MakeDict</span>**
```python
def MakeDict(self):
```
- **Description** : calls method ```ConstructGeometry``` and prints time statistics
- **Input** : /
- **Output** : /

