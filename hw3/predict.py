import sys
import os
import torch
import pandas as pd
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from src.model import ImprovedCNN

class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_paths = sorted([
            os.path.join(test_dir, img) for img in os.listdir(test_dir)
            if img.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('L')  # 灰階
        if self.transform:
            image = self.transform(image)
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        return image, img_id

def main():
    model_path = sys.argv[1]
    test_dir = sys.argv[2]
    output_csv = sys.argv[3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = ImprovedCNN(num_classes=7).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    results = []

    with torch.no_grad():
        for images, img_ids in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            preds = preds.cpu().numpy()
            results.append((preds[0]))

    print(f"Total predictions: {len(results)}")

    # 儲存成CSV
    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False, header=False)

if __name__ == "__main__":
    main()