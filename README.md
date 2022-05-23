# Fracture-Preprocessing-Code
Open-source preprocessing tool that can create, at the required level of accuracy, a fully conformal uniformly distributed grid for a given realistic fracture network. This leads to a robust way of constructing a hierarchy of Discrete-Fracture-Models for uncertainty quantification of energy production from reservoirs with natural fracture networks. Preprint of the paper related to this code can be found at: https://doi.org/10.1029/2021WR030743 .

An example of how to run the code is given in main.py 

If you want to preprocess your fracture network, make sure it is in the format:
  [[x1, y1, x2, y2]_1, [x1, y1, x2, y2]_2,...,[x1, y1, x2, y2]_n], where [x1, y1, x2, y2]_i are the x and y coordinates of the i-th fracture segment (defined by two points) and n is the total number of fracture segments.

In order to perform meshing you need to have gmsh installed and it must be recognized as an environment variable (i.e., you must be able to call it from the command line with "gmsh").

Only numpy, scipy, matplotlib, and igraph (and gmsh for meshing) are required to run this preprocessing tool.
