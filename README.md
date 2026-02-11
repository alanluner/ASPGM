# ASPGM
Adaptive Subgame Perfect Gradient Method

This code contains an implementation of the Adaptive Subgame Perfect Gradient Method introduced in 
> [Alan Luner and Benjamin Grimmer. "A Practical Adaptive Subgame Perfect Gradient Method", arXiv:2510.21617, 2025.](https://arxiv.org/abs/2510.21617).


The main algorithm is implemented in ASPGM.jl:
- First, construct your method with the desired settings:
  - To use default settings, set *method = ASPGM()*
  - To specify memory sizes k and t, set *method = ASPGM(k,t)*
  - To use BSPGM algorithm (no restarting or preconditioning) with memory size k, set *method = BSPGM(k)*

- Next, to run the algorithm with default settings, simply call *runMethod(method, oracle, x0)*, where oracle is a first-order oracle of the form f(x) = oracle(g,x), where g is updated in-place.

One can also use ASPGM11.jl to implement ASPGM-1-1 (memory size 1). This is a stand-alone implementation with no dependencies on external solvers.


## Experiments

Before running experiments, if you plan to run large-scale testing you will likely want to download additional datasets that we include in our results but are too large to include on github.
- From [LIBSVM](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/):
  - YearPredictionMSD (testing)
  - gisette
- From [LPFeas](https://plato.asu.edu/ftp/lptestset/):
  - brazil3
  - chromaticindex1024-7
  - ex10
  - graph40-40
  - qap15
  - rmine15
  - savsched1
  - scpm1
  - set-cover-model
  - supportcase10
  

We list below the executable scripts for reproducing results in our paper:
- run_SingleInstance_Synthetic - compare algorithm performance for a single synthetic problem instance, data is generated randomly as discussed in Section 4.2.
- run_SingleInstance_Real - compare algorithm performance for a single real-data problem instance from the LIBSVM or LPFeas datasets. See Figure 3 for an example.
- run_LargeScale_Synthetic - run algorithms for a large number of synthetic problem instances across a variety of problem classes, compare performance in terms of real-time and oracle calls. See Figure 1.
- run_LargeScale_Real - run algorithms over all real-data problems instances from LIBSVM or LPFeas datasets, compare performance in terms of real-time and oracle calls. See Figures 2 and 4.

## Package Requirements
- ASPGM-1-1
  - Parameters

- ASPGM
  - Mosek
  - MosekTools
  - JuMP
  - MathOptInterface
  - LinearAlgebra
  - Parameters

- Experiments
  - Random
  - LineSearches
  - LinearAlgebra
  - BlockDiagonals
  - SparseArrays
  - Optim
  - JLD2
  - QPSReader
  - Plots
  - LaTeXStrings

