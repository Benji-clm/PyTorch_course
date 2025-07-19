import os
from pathlib import Path
import requests
import zipfile
import torch
import torchvision
import random
import torchinfo
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from typing import Tuple, Dict, List

from helpers import download_files, find_classes, train, plot_loss_curves, eval_model

def main():
    print(f"Torch version: {torch.__version__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Setting device to: {device}")

    image_path = download_files()
    train_dir = image_path / "train"
    test_dir = image_path / "test"
    print(f"Training data directory: {train_dir} | Testing data directory: {test_dir}")

    train_transform_trivial_augment = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.TrivialAugmentWide(num_magnitude_bins=31),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor()
    ])

    train_data_augmented = datasets.ImageFolder(train_dir, transform=train_transform_trivial_augment)
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

    class TinyVGG(nn.Module):
        def __init__(self, input_shape: int, hidden_units: int, output_shape: int) -> None:
            super().__init__()
            self.conv_block_1 = nn.Sequential(
                nn.Conv2d(input_shape, hidden_units, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units, hidden_units, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2)
            )
            self.conv_block_2 = nn.Sequential(
                nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.conv_block_3 = nn.Sequential(
                nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(hidden_units, hidden_units, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(output_size=(1, 1)),  # outputs (B, C, 1, 1)
                nn.Flatten(),                              # -> (B, C)
                nn.Linear(hidden_units, output_shape)
            )

        def forward(self, x: torch.Tensor):
            return self.classifier(self.conv_block_3(self.conv_block_2(self.conv_block_1(x))))

    model = TinyVGG(input_shape=3,
                    hidden_units=128,
                    output_shape=len(train_data_augmented.classes)).to(device)
    
    print(next(model.parameters()).device) # sanity check that its on CUDA

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model_results = train(
        model=model,
        train_dataloader=train_dataloader_augmented,
        test_dataloader=test_dataloader_simple,
        optimizer=optimizer,
        device=device,
        loss_fn=loss_fn,
        epochs=100
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
    MODEL_PATH = Path("models")
    MODEL_PATH.mkdir(parents=True,
                     exist_ok=True)
    
    MODEL_NAME = "Reduced_Food101"
    MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

    print(f"Saving model to: {MODEL_SAVE_PATH}")
    torch.save(obj=model.state_dict(),
            f=MODEL_SAVE_PATH)
    


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()  # Safe for Windows devices (just use Linux Bro)
    main()
