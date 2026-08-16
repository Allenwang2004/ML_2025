import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

SAVE_PATH = "outputs/result_1.csv"

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
    test_data  = pd.read_csv("inputs/(additional_small)testing_dataset.csv", header=None).values.astype(np.float32)
    X_train, y_train = train_data[:, :2], train_data[:, 2]
    X_test, y_test = test_data[:, :2], test_data[:, 2]

    mse_records = []
    best_w = None
    best_mse = float("inf")

    grid_sizes = list(range(6, 50))
    sigmas = np.linspace(0.03, 0.2, 8)

    for grid_size in grid_sizes:
        for sigma in sigmas:
            grid_x = np.linspace(0, 1, grid_size)
            grid_y = np.linspace(0, 1, grid_size)
            centers = np.array([[x, y] for x in grid_x for y in grid_y])
            
            Phi_train = make_gaussian_basis(X_train, centers, sigma)
            Phi_test = make_gaussian_basis(X_test, centers, sigma)

            try:
                w = np.linalg.pinv(Phi_train.T @ Phi_train) @ Phi_train.T @ y_train
            except np.linalg.LinAlgError:
                continue

            y_pred = Phi_test @ w
            y_pred = np.clip(y_pred, 0, None)
            mse = np.mean((y_test - y_pred) ** 2)

            mse_records.append((grid_size, sigma, mse))

            print(f"Grid={grid_size}x{grid_size}, sigma={sigma:.3f}, MSE={mse:.2f}")

            if mse < best_mse:
                best_mse = mse
                best_grids = grid_size
                best_sigma = sigma
                best_w = w

    if best_w is not None:
        save_result(np.array([]), best_w)
        print(f"\nSaved weights to result_1.csv, MSE = {best_mse:.2f}, Grid={best_grids}x{best_grids}, sigma={best_sigma:.3f}")
    else:
        print(" No suitable model found (MSE ≥ 900)")

    # === 畫出 3D 圖 ===
    mse_df = pd.DataFrame(mse_records, columns=["grid_size", "sigma", "mse"])
    pivot_table = mse_df.pivot(index="grid_size", columns="sigma", values="mse")
    G, S = np.meshgrid(pivot_table.index, pivot_table.columns, indexing='ij')
    Z = pivot_table.values

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(G, S, Z, cmap='viridis')
    ax.set_xlabel('grid_size')
    ax.set_ylabel('sigma')
    ax.set_zlabel('MSE')
    ax.set_title('MSE Landscape (grid_size vs sigma)')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.tight_layout()
    plt.savefig("problem_1.png")

if __name__ == "__main__":
    main()