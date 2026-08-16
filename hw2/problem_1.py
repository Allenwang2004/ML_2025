# problem_1.py
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

def read_train_data():
    data_X = np.zeros((0, 28*28))
    data_T = np.zeros((0, 1), dtype=int)
    for i in range(10):
        for j in range(1000):
            img = np.array(Image.open(f"train/{i}/{j}.jpg"))
            data_X = np.vstack([data_X, img.reshape((1, 28*28))])
            data_T = np.vstack([data_T, i % 2])
    return data_X, data_T

def read_test_data():
    data_X = np.zeros((0, 28*28))
    for i in range(2000):
        img = np.array(Image.open(f"test/{i}.jpg"))
        data_X = np.vstack([data_X, img.reshape((1, 28*28))])
    return data_X

def save_result(y_pred):
    if not os.path.exists('./outputs'):
        os.makedirs('./outputs')
    with open('./outputs/result_1.csv', 'w') as f:
        for i in range(len(y_pred)):
            f.write(f"{y_pred[i, 0]}\n")

# 將z映射到(0, 1)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def train(X, y, epochs=1000, lr=0.01, lambda_l2=0.0):
    n_samples, n_features = X.shape
    W = np.zeros((n_features, 1))
    b = 0
    losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        z = np.dot(X, W) + b
        y_pred = sigmoid(z)

        error = y_pred - y

        dW = np.dot(X.T, error) / n_samples + lambda_l2 * W  # 加上 L2 項的梯度
        db = np.sum(error) / n_samples

        W -= lr * dW
        b -= lr * db

        # Loss 也要加上 L2懲罰項
        loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        l2_penalty = (lambda_l2 / 2) * np.sum(W * W)
        total_loss = loss + l2_penalty
        losses.append(total_loss)

    return W, b, losses

def cross_validation(X, y, folds=5, epochs=1000, lr=0.1, lambda_l2=0.0):
    fold_size = len(X) // folds
    accuracies = []

    for fold in range(folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]

        X_train_fold = np.vstack([X[:val_start], X[val_end:]])
        y_train_fold = np.vstack([y[:val_start], y[val_end:]])

        W, b, _= train(X_train_fold, y_train_fold, epochs=epochs, lr=lr, lambda_l2=lambda_l2)
        y_val_pred = predict(X_val, W, b)

        acc = (y_val_pred == y_val).mean()
        accuracies.append(acc)

    return np.mean(accuracies)

def predict(X, W, b):
    z = np.dot(X, W) + b
    y_pred = sigmoid(z)
    return (y_pred > 0.5).astype(int)

# 為了看是否有 overfitting
def train_with_validation(X, y, epochs=2000, lr=0.01, lambda_l2=0.0, val_ratio=0.2):
    n_samples, n_features = X.shape
    W = np.zeros((n_features, 1))
    b = 0

    idx = np.random.permutation(n_samples)
    split = int(n_samples * (1 - val_ratio))
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        z = np.dot(X_train, W) + b
        y_pred = sigmoid(z)

        error = y_pred - y_train

        dW = np.dot(X_train.T, error) / len(X_train) + lambda_l2 * W
        db = np.sum(error) / len(X_train)

        W -= lr * dW
        b -= lr * db

        train_loss = -np.mean(y_train * np.log(y_pred + 1e-8) + (1 - y_train) * np.log(1 - y_pred + 1e-8))
        train_loss += (lambda_l2 / 2) * np.sum(W * W)
        train_losses.append(train_loss)

        z_val = np.dot(X_val, W) + b
        y_val_pred = sigmoid(z_val)

        val_loss = -np.mean(y_val * np.log(y_val_pred + 1e-8) + (1 - y_val) * np.log(1 - y_val_pred + 1e-8))
        val_loss += (lambda_l2 / 2) * np.sum(W * W)
        val_losses.append(val_loss)

    return W, b, train_losses, val_losses


if __name__ == "__main__":
    X_train, y_train = read_train_data()
    X_test = read_test_data()

    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # For cv
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    learning_rates = [0.55, 0.56, 0.57, 0.58, 0.59]
    lambda_l2_list = [1e-5, 1e-4, 1e-3, 1e-2]

    best_lr = None
    best_lambda = None
    best_acc = 0

    for lr in learning_rates:
        for lambda_l2 in lambda_l2_list:
            avg_acc = cross_validation(X_train, y_train, folds=5, epochs=1000, lr=lr, lambda_l2=lambda_l2)
            print(f"Learning rate {lr}, Lambda {lambda_l2}, Cross-Validation Accuracy: {avg_acc:.4f}")

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_lr = lr
                best_lambda = lambda_l2

    print(f"Best Learning Rate: {best_lr}, Best Lambda: {best_lambda}, Accuracy: {best_acc:.4f}")

    W, b, losses = train(X_train, y_train, epochs=1000, lr=best_lr, lambda_l2=best_lambda)

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('./loss_curve_1.png')

    y_pred = predict(X_test, W, b)
    save_result(y_pred)

    W, b, train_losses, val_losses = train_with_validation(X_train, y_train, epochs=1000, lr=best_lr, lambda_l2=best_lambda)
    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./loss_curve1_with_val.png')


