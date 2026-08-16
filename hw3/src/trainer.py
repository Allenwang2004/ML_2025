import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset import FacialExpressionDataset, load_data
from model import ImprovedCNN
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np


def plot_loss_acc(train_losses, valid_losses, valid_accs):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(valid_losses, label="Valid Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(valid_accs, label="Valid Accuracy", color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig("loss_acc_curve.png")
    plt.close()


def plot_confusion(y_true, y_pred, filename='conf_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title("Confusion Matrix")
    plt.savefig(filename)
    plt.close()


def train(model, train_loader, valid_loader, device, num_epochs=30, lr=1e-3, save_path="model.pth"):
    # Cost-sensitive weights (fear: 2, sad: 4 boosted)
    class_weights = torch.tensor([
        0.9890,  # angry
        0.9698,  # disgust
        1.5730,  # fear (boosted)
        0.9770,  # happy
        1.4959,  # sad (boosted)
        1.0042,  # surprise
        1.0183   # neutral
    ], dtype=torch.float).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    train_losses = []
    valid_losses = []
    valid_accs = []
    best_valid_loss = float('inf')
    patience = 7
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        train_losses.append(train_loss)

        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        valid_bar = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Valid]")
        with torch.no_grad():
            for images, labels in valid_bar:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                loss = criterion(outputs, labels)

                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                all_preds.append(predicted.cpu())
                all_labels.append(labels.cpu())
                valid_bar.set_postfix(loss=loss.item())

        valid_loss = running_loss / len(valid_loader)
        valid_losses.append(valid_loss)
        valid_acc = 100 * correct / total
        valid_accs.append(valid_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {train_loss:.4f} Valid Loss: {valid_loss:.4f} Valid Acc: {valid_acc:.2f}%")

        scheduler.step(valid_loss)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), save_path)
            print(f"Saved Best Model at Epoch {epoch+1}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"⏹ Early stopping at epoch {epoch+1}")
                break

    # Plot and save
    plot_loss_acc(train_losses, valid_losses, valid_accs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    plot_confusion(all_labels, all_preds, 'conf_matrix.png')
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, digits=4))


def main():
    num_epochs = 50
    batch_size = 64
    learning_rate = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_list, valid_list = load_data("balanced_data.csv")

    train_dataset = FacialExpressionDataset(train_list, transform=transform)
    valid_dataset = FacialExpressionDataset(valid_list, transform=transform)

    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = ImprovedCNN(num_classes=7).to(device)
    summary(model, input_size=(1, 48, 48))

    train(model, train_loader, valid_loader, device, num_epochs=num_epochs, lr=learning_rate, save_path="../model.pth")


if __name__ == "__main__":
    main()