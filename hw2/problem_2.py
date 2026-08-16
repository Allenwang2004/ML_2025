# problem_2.py
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
            data_T = np.vstack([data_T, i])
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
    with open('./outputs/result_2.csv', 'w') as f:
        for i in range(len(y_pred)):
            f.write(f"{y_pred[i]}\n")

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

def one_hot(y, num_classes):
    return np.eye(num_classes)[y.reshape(-1)]

def train(X, y, epochs=1000, lr=0.1, lambda_l2=0.0, num_classes=10):
    n_samples, n_features = X.shape
    W = np.zeros((n_features, num_classes))
    b = np.zeros((1, num_classes))

    y_onehot = one_hot(y, num_classes)
    losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        z = np.dot(X, W) + b
        y_pred = softmax(z)

        error = y_pred - y_onehot

        dW = np.dot(X.T, error) / n_samples + lambda_l2 * W
        db = np.sum(error, axis=0, keepdims=True) / n_samples

        W -= lr * dW
        b -= lr * db

        loss = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-8), axis=1))
        l2_penalty = (lambda_l2 / 2) * np.sum(W * W)
        total_loss = loss + l2_penalty
        losses.append(total_loss)

    return W, b, losses

def cross_validation(X, y, folds=5, epochs=2000, lr=0.1, lambda_l2=0.0, num_classes=10):
    fold_size = len(X) // folds
    accuracies = []

    for fold in range(folds):
        val_start = fold * fold_size
        val_end = val_start + fold_size

        X_val = X[val_start:val_end]
        y_val = y[val_start:val_end]

        X_train_fold = np.vstack([X[:val_start], X[val_end:]])
        y_train_fold = np.vstack([y[:val_start], y[val_end:]])

        W, b, _ = train(X_train_fold, y_train_fold, epochs=epochs, lr=lr, lambda_l2=lambda_l2, num_classes=num_classes)
        y_val_pred = predict(X_val, W, b)

        acc = (y_val_pred.reshape(-1) == y_val.reshape(-1)).mean()
        accuracies.append(acc)

    return np.mean(accuracies)

def predict(X, W, b):
    z = np.dot(X, W) + b
    y_pred = softmax(z)
    return np.argmax(y_pred, axis=1)

def train_with_validation(X, y, epochs=1000, lr=0.1, lambda_l2=0.0, val_ratio=0.2, num_classes=10):
    n_samples, n_features = X.shape
    W = np.zeros((n_features, num_classes))
    b = np.zeros((1, num_classes))

    idx = np.random.permutation(n_samples)
    split = int(n_samples * (1 - val_ratio))
    X_train, X_val = X[idx[:split]], X[idx[split:]]
    y_train, y_val = y[idx[:split]], y[idx[split:]]

    y_train_onehot = one_hot(y_train, num_classes)
    y_val_onehot = one_hot(y_val, num_classes)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs), desc="Training"):
        z = np.dot(X_train, W) + b
        y_pred = softmax(z)

        error = y_pred - y_train_onehot

        dW = np.dot(X_train.T, error) / len(X_train) + lambda_l2 * W
        db = np.sum(error, axis=0, keepdims=True) / len(X_train)

        W -= lr * dW
        b -= lr * db

        train_loss = -np.mean(np.sum(y_train_onehot * np.log(y_pred + 1e-8), axis=1))
        train_loss += (lambda_l2 / 2) * np.sum(W * W)
        train_losses.append(train_loss)

        z_val = np.dot(X_val, W) + b
        y_val_pred = softmax(z_val)
        val_loss = -np.mean(np.sum(y_val_onehot * np.log(y_val_pred + 1e-8), axis=1))
        val_loss += (lambda_l2 / 2) * np.sum(W * W)
        val_losses.append(val_loss)

    return W, b, train_losses, val_losses

if __name__ == "__main__":
    X_train, y_train = read_train_data()
    X_test = read_test_data()

    # Normalize
    X_train = X_train / 255.0
    X_test = X_test / 255.0

    # Shuffle
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    learning_rates = [0.5, 0.52, 0.54, 0.56, 0.58, 0.6]
    lambda_l2_list = [1e-5, 1e-4, 1e-3, 1e-2]
    best_lr = None
    best_lambda = None
    best_acc = 0

    for lr in learning_rates:
        for lambda_l2 in lambda_l2_list:
            avg_acc = cross_validation(X_train, y_train, folds=5, epochs=1000, lr=lr, lambda_l2=lambda_l2, num_classes=10)
            print(f"Learning rate {lr}, Lambda {lambda_l2}, Cross-Validation Accuracy: {avg_acc:.4f}")

            if avg_acc > best_acc:
                best_acc = avg_acc
                best_lr = lr
                best_lambda = lambda_l2

    print(f"Best Learning Rate: {best_lr}, Best Lambda: {best_lambda}, Accuracy: {best_acc:.4f}")


    W, b, losses = train(X_train, y_train, epochs=1000, lr=best_lr, lambda_l2=best_lambda, num_classes=10)

    plt.plot(losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.grid(True)
    plt.savefig('./loss_curve_2.png')

    y_pred = predict(X_test, W, b)
    save_result(y_pred)

    W, b, train_losses, val_losses = train_with_validation(X_train, y_train, epochs=1000, lr=best_lr, lambda_l2=best_lambda, num_classes=10)

    plt.figure()
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('./loss_curve2_with_val.png')


