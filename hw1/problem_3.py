import numpy as np
import pandas as pd
import os

SAVE_PATH = "outputs/result_3.csv"

def save_result(preds: np.ndarray, weights: np.ndarray):
    max_length = max(len(preds), len(weights))
    result = np.full((max_length, 2), "", dtype=object)
    result[:len(preds), 0] = preds.astype(str)
    result[:len(weights), 1] = weights.astype(str)
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    np.savetxt(SAVE_PATH, result, delimiter=",", fmt="%s")

def make_gaussian_basis(X, centers, sigma):
    Phi = []
    for mu in centers:
        basis = np.exp(-np.sum((X - mu) ** 2, axis=1) / (2 * sigma ** 2))
        Phi.append(basis)
    return np.stack(Phi, axis=1)

def main():
    train_data = pd.read_csv("inputs/training_dataset.csv", header=None).values.astype(np.float32)
    test_data = pd.read_csv("inputs/(additional_small)testing_dataset.csv", header=None).values.astype(np.float32)

    X_train, y_train = train_data[:, :2], train_data[:, 2]
    X_test, y_test = test_data[:, :2], test_data[:, 2]

    best_mse = float("inf")
    best_weights = None
    best_pred = None
    best_config = None

    for grid_size in range(6, 40):
        for sigma in np.linspace(0.03, 0.2, 5):
            for alpha in [0.1, 1.0, 10.0]:
                for beta in [10.0, 50.0, 100.0]:
                    
                    grid_x = np.linspace(0, 1, grid_size)
                    grid_y = np.linspace(0, 1, grid_size)
                    centers = np.array([[x, y] for x in grid_x for y in grid_y])

                    Phi_train = make_gaussian_basis(X_train, centers, sigma)
                    Phi_test = make_gaussian_basis(X_test, centers, sigma)
                    I = np.eye(Phi_train.shape[1])

                    SN_inv = alpha * I + beta * Phi_train.T @ Phi_train
                    try:
                        SN = np.linalg.inv(SN_inv)
                    except np.linalg.LinAlgError:
                        continue
                    mN = beta * SN @ Phi_train.T @ y_train

                    y_pred = Phi_test @ mN
                    y_pred = np.clip(y_pred, 0, None)
                    mse = np.mean((y_test - y_pred) ** 2)

                    print(f"Grid={grid_size}x{grid_size}, sigma={sigma:.3f}, alpha={alpha}, β={beta}, MSE={mse:.2f}")

                    if mse < best_mse:
                        best_mse = mse
                        best_weights = mN
                        best_pred = np.clip(Phi_test @ mN, 0, None)
                        best_config = (grid_size, sigma, alpha, beta)

    if best_weights is not None:
        save_result(best_pred, best_weights)
        g, s, a, b = best_config
        print(f"\n Done! Best MSE={best_mse:.2f}")
    else:
        print("Failed: no config with MSE < 900")

if __name__ == "__main__":
    main()
