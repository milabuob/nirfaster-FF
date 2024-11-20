This is the source code for compiling the `cgalmesher*` binaries, which generates tetrahedral meshes from segmented volumes.

## How to use

To use them, the syntax is,

`cgalmesher* volume.inr result.mesh [criteria.txt]`

where "volume.inr" is the segmented volume to be meshed, saved in the INRIA format, "result.mesh" is the name of the file containing the meshing result (MEDIT format), and optionally, "criteria.txt" specifies the meshing parameters, which is organized as,

```
facet_angle
facet_size
facet_distance
cell_radius_edge
general_cell_size
bool_lloyd_smooth
num_subdomains
label1 cellsize1
label2 cellsize2
...
```

If `num_subdomains=0`, the subsequent lines should be empty. The detailed meanings of the parameters are documented here: [CGAL 6.0.1 - 3D Mesh Generation: User Manual](https://doc.cgal.org/latest/Mesh_3/index.html#Chapter_3D_Mesh_Generation) If the file is not provided, the default parameters are used, which is equivalent to the following file,

```
25.0
2.0
1.5
2.0
2.0
0
0
```

Suppose we want regions labeled 1 to have cell size of 1.5, regions labeled 3 to have cell size 2.5, and all other regions to have cell size of *general_cell_size*, the file should be, if all other parameters are kept the same,

```
25.0
2.0
1.5
2.0
2.0
0
2
1 1.5
3 2.5
```

## Licensing

This code as well as the resulting binaries (namely, *cgalmesherLINUX*, *cgalmesherMAC*, and *cgalmesher.exe*) utilizes the [CGAL 6.0.1 library](https://www.cgal.org/), whose components are licensed under either GPL v3+ or LGPL v3+. This means that the mesher binaries, as well as the NIRFASTerFF function `nirfasterff.meshing.meshutils.RunCGALMeshGenerator()` which uses the binaries through a Python system call, may be restricted from commercial use. 

The original licenses from the CGAL library can also be found in this folder.

## Compiling

On Linux, the binary was compiled using

```bash
g++ -O3 -std=c++17 -DCGAL_DISABLE_GMP=1 -DCMAKE_OVERRIDDEN_DEFAULT_ENT_BACKEND=3 -I/path/to/CGAL-6.0.1/include -I/path/to/boost_1_76_0 cgal_mesher.cpp -o cgalmesherLINUX
```

On Mac, the same command was used twice, for both ARM and Intel architectures, and the final fat binary was created using

```bash
lipo cgalmesherARM cgalmesherINTEL -create -output cgalmesherMAC
```

On Windows, the binary was compiled using

```cmd
cl.exe /O2 /std:c++17 /I"path\to\CGAL-6.0.1\include" /I"path\to\boost_1_76_0" /D CGAL_DISABLE_GMP=1 /D CMAKE_OVERRIDDEN_DEFAULT_ENT_BACKEND=3 /EHsc cgal_mesher.cpp /Fecgalmesher.exe
```


