from skopt import gp_minimize
from skopt.space import Real, Integer
from skopt.utils import use_named_args
import torch
from torch.utils.data import DataLoader
from dataset import FacialExpressionDataset, load_data
from model import ImprovedCNN
from train import train
from torchvision import transforms

# 定義超參數空間
space  = [
    Real(1e-5, 1e-2, name='learning_rate', prior='log-uniform'),
    Integer(32, 128, name='batch_size'),
    Real(1e-6, 1e-2, name='weight_decay', prior='log-uniform')
]

@use_named_args(space)
def objective(**params):
    print("\nTrial with params:", params)

    # 設定裝置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 載入資料集
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_list, valid_list = load_data("balanced_data.csv")
    train_dataset = FacialExpressionDataset(train_list, transform=transform)
    valid_dataset = FacialExpressionDataset(valid_list, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=params['batch_size'], shuffle=False)

    model = ImprovedCNN(num_classes=7).to(device)

    # 執行訓練並回傳 validation loss 作為目標
    _, valid_loss = train(
        model, train_loader, valid_loader, device,
        num_epochs=10,
        lr=params['learning_rate'],
        weight_decay=params['weight_decay'],
        save_path="temp.pth",
        return_loss_only=True
    )
    return valid_loss

if __name__ == "__main__":
    res = gp_minimize(objective, space, n_calls=20, random_state=42)

    print("\nBest hyperparameters:")
    print(f"Learning rate: {res.x[0]:.6f}")
    print(f"Batch size: {res.x[1]}")
    print(f"Weight decay: {res.x[2]:.6f}")
