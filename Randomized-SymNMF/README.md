# Randomized SymNMF

This repo contains the core MATLAB code for randomized NMF algorithms.
The paper associated with this code can be found at : 
https://arxiv.org/pdf/2402.08134
"RANDOMIZED ALGORITHMS FOR SYMMETRIC NONNEGATIVE MATRIX FACTORIZATION" to
appear in the SIAM Journal of Matrix Analysis and Applications (2024). 

The functions nnlsm_blockpivot and rand_index are borrowed functions from
existing sources and not authored as part of this repo.
Details can be found in their respective files.

The main functions of interest are LAI_NMF, LAI_SymPGNCG, and lvs_symNMF.
These functions implement Low-rank Approximate Input NMF with varoius update functions
available and leverage score sampling for SymNMF.
The LAI-NMF implementations support a variety of algorithms for NMF and
SymNMF using HALS and BPP update rules.
For SymNMF the leverage score sampling and PGNCG methods are also available.

A sample of running these functions can be found in getNMF_Problem.m
