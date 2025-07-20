"""
Trains a PyTorch image classification model using device-agnostic code.
"""

import os
import torch
import data_setup, engine, model_builder, utils
import argparse

from torchvision import transforms

# Argparse method for setting hyperparameters

parser = argparse.ArgumentParser(
  prog='train.py',
  description='Train a custom ResNet model on a Food101 subset',
  epilog='Rip my GPU'
)

parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
# parser.add_argument('--hidden_units', type=int, default=10, help='Number of hidden units within layers')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--model_name', type=str, default='models/Reduced_Food101_resnet', help='Path to save or load model')

def main():

  args = parser.parse_args()

  # Setup hyperparameters
  NUM_EPOCHS = args.epochs
  BATCH_SIZE = args.batch_size
  # HIDDEN_UNITS = args.hidden_units
  LEARNING_RATE = args.lr
  MODEL_NAME= args.model_name

  # Setup directories
  image_path = utils.get_food101_subset()
  train_dir = image_path / "train"
  print(f"train_dir: {train_dir}")
  test_dir = image_path / "test"

  # Setup target device
  device = "cuda" if torch.cuda.is_available() else "cpu"

  # Create transforms
  data_transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor()
  ])

  # Create DataLoaders with help from data_setup.py
  train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
      train_dir=train_dir,
      test_dir=test_dir,
      transform=data_transform,
      batch_size=BATCH_SIZE
  )

  # Create model with help from model_builder.py
  model = model_builder.ResNetCustom(
      input_channels=3,
      num_classes=len(class_names)
  ).to(device)

  # Set loss and optimizer
  loss_fn = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(),
                              lr=LEARNING_RATE)

  # Start training with help from engine.py
  engine.train(model=model,
              train_dataloader=train_dataloader,
              test_dataloader=test_dataloader,
              loss_fn=loss_fn,
              optimizer=optimizer,
              epochs=NUM_EPOCHS,
              device=device)

  # Save the model with help from utils.py
  utils.save_model(model=model,
                  model_name=MODEL_NAME)


if __name__ == "__main__":
  main()
