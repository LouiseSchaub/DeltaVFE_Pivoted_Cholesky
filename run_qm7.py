from core import run_full_experiment
from hyperparameters import load_hypers


def main():
    s_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    for fold in range(5):
        ell, sigma2 = load_hypers(fold=fold)
        run_full_experiment(dataset="QM7", fold=fold, ell=ell, sigma2=sigma2, m=2048, s_values=s_values)


if __name__ == "__main__":
    main()