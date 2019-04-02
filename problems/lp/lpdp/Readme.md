# LPDP algorithm for Longest Path Problem

LPDP algoritm solves Longest Path (LP) problem and is one of the SOTA for this problem. 

[LPDP Homepage](http://algo2.iti.kit.edu/kalp/index.html) describes the algorithm and provides link to the code,
datasets, and description. 

```
@techreport{balyo2017, 
             AUTHOR = {Balyo, Tomas and Fieger, Kai and Schulz, Christian},
             TITLE = {{Optimal Longest Paths by Dynamic Programming}},
             PUBLISHER = {Springer},
             JOURNAL = {Technical Report. arXiv:1702.04170},
             YEAR = {2017}
}
```

## Dependencies

* Unix-like system
* Python 3
* g++

## Description

LPDP is written in C++ and requires several packages to be installed (`argtable2`, `tbb`, `scons`). 
File `lpdp_baseline.py` allows one to install all the necessary dependencies and run the code. 
In particular, during the first run it will install `argtable2`, `tbb`, `scons` locally, with no administrative rights.
It make take a while (5-10 mins), since it downloads and compiles the packages, but is done only once. 
You should get three folders, `argtable2`, `scons`, and `tbb` in the folder `lpdp`, then your installation went OK.
If you are successful, the code will proceed to update environment variables locally, and then run
compiled code. Also this code has function `download_lpdp_datasets()`, which will download datasets.
Note datasets are big, in compressed form they are 2GB and after uncompression they are 12GB. 
Function `download_lpdp_datasets()` will download only datasets if they don't exist yet. 

## Quick start

For many experiments, it's easier to set up all arguments in the file directly.

```python
cwd = os.path.abspath(os.path.join("lpdp"))
kalp = os.path.join(cwd, 'kalp')
graph_fn = f"{kalp}/examples/Grid8x8.graph"
start_vertex = 0
for target_vertex in range(1, 25):
    run_kalp(graph_fn, start_vertex, target_vertex, output_filename='test.txt',
            results_filename='results.txt')
```

This will write LPDP (kalp executable) on graph `Grid8x8.graph` with `0`-th vertex to vertices from 
`1` to `24`, save longest path to `test.txt` and then append setup, runtime, and length of longest
path to file `results.txt` for each run.  
