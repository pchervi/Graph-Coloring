# Graph-Coloring
I have been interested for some times in setting up a Graph Coloring program to color graph vertices.

The Python code used PySCIPOpt, the Python wrapper of SCIP, and examples from SCIP or PySCIPOpt documentation such as a former 'test_coloring.py' example and associated C++ SCIP code. Many thanks to the SCIP communauty for paving the way to thos code.

Graph Coloring uses branch-and-price and column generation to find an optimal coloring solution. It also relies on a greedy heuristic and an exact heuristic to find minimum weighted independent sets (MWIS), a greedy and a Tabucol routine to find an intial coloring.

Graphs with up to 100 vertices can be processed, although performance rapidly decreased as number of vertices increases and graph density decreases.

I would appreciate any feedback to improve the code, any remark is welcome!
