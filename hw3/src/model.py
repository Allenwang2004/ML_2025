import torch
import torch.nn as nn

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=7):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Dropout(0.25)
            )

        self.conv = nn.Sequential(
            conv_block(1, 64),    # 48x48 → 24x24
            conv_block(64, 128),  # 24x24 → 12x12
            conv_block(128, 256), # 12x12 → 6x6
            conv_block(256, 512), # 6x6 → 3x3
        )

        # Global Average Pooling → (B, 512)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.gap(x)
        x = self.classifier(x)
        return x