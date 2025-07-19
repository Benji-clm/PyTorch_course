import os
from pathlib import Path
import requests
import zipfile
import torch
import torchvision
import random
import torchinfo
import matplotlib.pyplot as plt
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import Food101
from typing import Tuple, Dict, List

from helpers import download_files, find_classes, train, plot_loss_curves, eval_model, prepare_food101_subset, get_food101_subset

def main():
    print(f"Torch version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting device to: {device}")

    image_path = get_food101_subset()
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    print(f"Training data directory: {train_dir} | Testing data directory: {test_dir}")

    normalize = transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(128, scale=(0.8,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(128),
        transforms.ToTensor(),
        normalize
    ])

    train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform)
    test_data_simple = datasets.ImageFolder(test_dir, transform=test_transform)

    BATCH_SIZE = 16
    # NUM_WORKERS = os.cpu_count()
    NUM_WORKERS = 0

    train_dataloader_augmented = DataLoader(
        dataset=train_data_augmented,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=True
    )

    test_dataloader_simple = DataLoader(
        test_data_simple,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        shuffle=False
    )

    class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, downsample=False):
            super().__init__()
            stride = 2 if downsample else 1

            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

            self.downsample = nn.Sequential()
            if downsample or in_channels != out_channels:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm2d(out_channels)
                )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            identity = self.downsample(x)
            out = self.conv_block(x)
            return self.relu(out + identity)


    class ResNetCustom(nn.Module):
        def __init__(self, input_channels: int, num_classes: int):
            super().__init__()

            self.stem = nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            )

            self.layer1 = ResidualBlock(64, 64)
            self.layer2 = ResidualBlock(64, 128, downsample=True)
            self.layer3 = ResidualBlock(128, 256, downsample=True)
            self.layer4 = ResidualBlock(256, 512, downsample=True)

            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        def forward(self, x):
            x = self.stem(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            return self.classifier(x)

    
    # Loading model if it already exists
    MODEL_PATH = Path(__file__).parent / "models"
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)
    MODEL_NAME = "Reduced_Food101_resnet"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    model = ResNetCustom(input_channels=3, num_classes=len(train_data_augmented.classes)).to(device)
    
    if MODEL_SAVE_PATH.exists():
        print(f"Model already exists, loading {MODEL_SAVE_PATH}")
        model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

    
    print(next(model.parameters()).device) # sanity check that its on CUDA

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    model_results = train(
        model=model,
        train_dataloader=train_dataloader_augmented,
        test_dataloader=test_dataloader_simple,
        optimizer=optimizer,
        device=device,
        scheduler=scheduler,
        loss_fn=loss_fn,
        epochs=20
    )

    plot_loss_curves(model_results)

    model_results = eval_model(
        model=model,
        data_loader=test_dataloader_simple,
        loss_fn=loss_fn,
        device=device
    )

    print(f"Model results: {model_results}")


    # Saving the model (create directory if needed)
    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
            f=MODEL_SAVE_PATH)
    


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Safe for Windows devices (just use Linux Bro)
    main()
