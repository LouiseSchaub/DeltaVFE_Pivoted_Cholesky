# $\Delta$-VFE Pivoted Cholesky 
This is the accompanying repository for the paper "Variational Free Energy Pivot Selection for Pivoted Cholesky" by [Louise Schaub](https://louiseschaub.github.io) and [Peter Zaspel](https://peter-zaspel.de).
You can find the preprint [here]().

## Necessary Libraries 
For the computations found in the paper, we have used `Python 3.13.11`. In order to execute the code included in this repository, you will need to install:
- `NumPy (2.3.5)`
- `SciPy (1.17.0)`
- `Pandas (3.0.0)`
- `scikit-learn (1.8.0)`
- `tqdm (4.67.1)` 
- `ucimlrepo (0.07)` 

For plotting you will need in addition
- `matplotlib (3.10.8)`
- `seaborn (0.13.2)`

The specific version that were used to generate the results are given in brackets.
 
## Structure of the repository
This repository includes the code to: 
- Reproduce the results and 
- figures that can be found in the paper. 
- Compute approximate Cholesky factors using classical pivoted Cholesky, RPCholesky and $\Delta$-VFE pivoted Cholesky. Where the first two are used for comparison.

### Included Functions
We list the included functions alphabetically and give a short overview of their usage.
- `core.py`: Core functions to produce the results given in the paper. 
- `data_handling.py`: Loads and prepares the datasets (Abalone, QM7).
- `delta_vfe.py`: Realization of the $\Delta$-VFE Pivoted Cholesky algorithm as found in our paper (batched version).
- `full_gpr.py`: Creates the baseline reference, i.e. runs GPR on the full (training) dataset using the original kernel matrix.
- `greedy_cholesky.py`: Realization of classical pivoted Cholesky.
- `helpers.py`: Several helper functions. 
- `hyperparameters.py`: Computation and handling of hyperparameters. 
- `kernels.py`: Used to construct kernel matrices, i.e. Gaussian and Laplacian kernel matrices. 
- `main.py`: Runs the experiments on the available dataset.
- `metrics.py`: Includes the metrics of interest, i.e. predicitve errors (RMSE, MAE), VFE evaluation, predictive mean.
- `rpcholesky.py`: Realization of randomly pivoted Cholesky.
- `run_abalone.py`: Runs the experiment on the Abalone dataset.
- `run_qm7.py`: Runs the experiment on the QM7 dataset.
In order to reproduce the plots found in the paper, we have included the following files:
- `plot_vfe_approximation.py`: Plots 
- `plot_prediction.py`:
- `plot_s_comparison`: 
- `plot_utils.py`: utility functions used for plotting. This is shared across all plotting routines.
### Included Data
We have included all the data that has been used for Section 7 of our paper. The data is organized in the following way:
`Data/{DATASET}/{METHOD}/{METHOD}_kernelwidth={KERNELWIDTH}_noise2={NOISE_VARIANCE2}_m={TARGET_RANK}.csv`. For $\Delta$-VFE there is an additional `_s={BATCH_SIZE}` in the end.

### Hyperparameters & Baseline Reference Values
The hyperparameters used for the respective experiments, can be found in the corresponding folder `Data/Abalone/Full/full_gaussian_hypers.dat` and 
`Data/QM7/Full/full_laplacian_fold={FOLD}_hypers.dat`, where FOLD stands for one of QM7's five folds. Additionally, these files contain the 
MAEs and RMSEs (original and standardized) of the original problem, i.e. using the entire (training) dataset, as well as the seeds used for the experiments. 

### The $\Delta$-VFE Pivoted Cholesky Algorithm
... is included in `delta_vfe.py` and is implemented in a batched manner. The file includes all necessary helpers to use the method $\Delta$-VFE pivoted Cholesky can be called as 
```
delta_vfe_pivots_batched(K, y, m, sigma2, s)
```
where `K`: kernel matrix $K$, y: output data vector $\mathbf{y}$, `m`: target-rank $r$, `sigma2`:squared noise variance $\sigma_\varepsilon^2$ and `s`: batch factor $s$.
This routine currently returns only the list of pivotts. If you want the approximate Cholesky factor $L$ such that $K \approx LL^\top$, add `L` to the return-statement.

## Reproducing the Results from the Paper
After having installed all the necessary packages, you can easily reproduce the results found in the paper, by simply using `main.py`. 
This will generate the results for `Abalone` and `QM7` datasets. If you want to run a specific dataset, then execute `run_abalone.py` or `run_qm7.py`.
There you can further specify `s_values`, which is a list of batch sizes $s$, that will be used for $\Delta$-VFE pivoted Cholesky.
