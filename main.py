"""
Run this file to reproduce the results found in the paper "Variational Free Energy Pivot Selection for Pivoted Cholesky".
If you wish to make changes to the parameters, go to run_abalone.py and rum_qm7.py.
"""

import run_abalone
import run_qm7

if __name__ == "__main__":
    print("\n" + "#" * 42)
    print("# Abalone experiments")
    print("#" * 42)
    run_abalone.main()

    print("\n" + "#" * 42)
    print("# QM7 experiments (5 folds)")
    print("#" * 42)
    run_qm7.main()

    print("\nDone.")