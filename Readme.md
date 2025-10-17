This repository contains the reference implementation for **SNMPBB**: a fast, projected-gradient-based algorithm that extends the Nonmonotone Projected Barzilai–Borwein (NMPBB) method to symmetric nonnegative matrix factorization case and graph clustering tasks.

SNMPBB combines:
- Projected gradient descent with Barzilai–Borwein step size selection,
- A nonmonotone line search for faster convergence,
- A symmetric penalty term to enforce $V ≈ WW^T$,
- Optional **graph regularization** (Laplacian + sparsity) to improve clustering quality,
- Low-rank approximate input (LAI) acceleration for large-scale problems.

## Installation

Clone the repo:

```bash
git clone https://github.com/yourusername/SNMPBB.git
cd SNMPBB
```
