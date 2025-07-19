import os
from pathlib import Path
import requests
import zipfile
import torch
import torchvision
import random
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import Food101
from typing import Tuple, Dict, List
from tqdm.auto import tqdm
import matplotlib.pyplot as plt


def download_files() -> str:
    # Setup path to data folder
    data_path = Path("data/")
    image_path = data_path / "pizza_steak_sushi"

    # If the image folder doesn't exist, download it and prepare it...
    if image_path.is_dir() and os.listdir(image_path) != 0:
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

    # download pizza, steak and sushi
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading images data")
        f.write(request.content)

    # Unzip data
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip" , "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data...")
        zip_ref.extractall(image_path)

    return image_path



def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """Finds the class folder names in a target directory.

    Assumes target directory is in standard image classification format.

    Args:
        directory (str): target directory to load classnames from.

    Returns:
        Tuple[List[str], Dict[str, int]]: (list_of_class_names, dict(class_name: idx...))

    Example:
        find_classes("food_images/train")
        >>> (["class_1", "class_2"], {"class_1": 0, ...})
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
      raise FileNotFoundError(f"Couldn't find any classes in {directory}.")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx



def train_step(model: torch.nn.Module, 
               dataloader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    # Put model in train mode
    model.train()
    
    # Setup train loss and train accuracy values
    train_loss, train_acc = 0, 0
    
    # Loop through data loader data batches
    for batch, (X, y) in enumerate(dataloader):
        # Send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate  and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item() 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += (y_pred_class == y).sum().item()/len(y_pred)

    # Adjust metrics to get average loss and accuracy per batch 
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)
    return train_loss, train_acc



def test_step(model: torch.nn.Module, 
              dataloader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module,
              device: torch.device):
    # Put model in eval mode
    model.eval() 
    
    # Setup test loss and test accuracy values
    test_loss, test_acc = 0, 0
    
    # Turn on inference context manager
    with torch.inference_mode():
        # Loop through DataLoader batches
        for batch, (X, y) in enumerate(dataloader):
            # Send data to target device
            X, y = X.to(device), y.to(device)
    
            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()
            
            # Calculate and accumulate accuracy
            test_pred_labels = test_pred_logits.argmax(dim=1)
            test_acc += ((test_pred_labels == y).sum().item()/len(test_pred_labels))
            
    # Adjust metrics to get average loss and accuracy per batch 
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)
    return test_loss, test_acc


def train(model: torch.nn.Module, 
          train_dataloader: torch.utils.data.DataLoader, 
          test_dataloader: torch.utils.data.DataLoader, 
          optimizer: torch.optim.Optimizer,
          device: torch.device,
          loss_fn: torch.nn.Module = nn.CrossEntropyLoss(),
          epochs: int = 5):
    
    # 2. Create empty results dictionary
    results = {"train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # 3. Loop through training and testing steps for a number of epochs
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer,
                                           device=device)
        
        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn,
                                        device=device)
        
        # 4. Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # 5. Update results dictionary
        # Ensure all data is moved to CPU and converted to float for storage
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)

    # 6. Return the filled results at the end of the epochs
    return results


def plot_loss_curves(results: Dict[str, List[float]]):
    """Plots training curves of a results dictionary.

    Args:
        results (dict): dictionary containing list of values, e.g.
            {"train_loss": [...],
             "train_acc": [...],
             "test_loss": [...],
             "test_acc": [...]}
    """
    
    # Get the loss values of the results dictionary (training and test)
    loss = results['train_loss']
    test_loss = results['test_loss']

    # Get the accuracy values of the results dictionary (training and test)
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']

    # Figure out how many epochs there were
    epochs = range(len(results['train_loss']))

    # Setup a plot 
    plt.figure(figsize=(15, 7))

    # Plot loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, test_loss, label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs, test_accuracy, label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend();

    plt.tight_layout()
    plt.show


def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to predictions.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def eval_model(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               device: torch.device,
               accuracy_fn = accuracy_fn):
    """Evaluates a given model on a given dataset.

    Args:
        model (torch.nn.Module): A PyTorch model capable of making predictions on data_loader.
        data_loader (torch.utils.data.DataLoader): The target dataset to predict on.
        loss_fn (torch.nn.Module): The loss function of model.
        accuracy_fn: An accuracy function to compare the models predictions to the truth labels.
        device (str, optional): Target device to compute on. Defaults to device.

    Returns:
        (dict): Results of model making predictions on data_loader.
    """


    loss, acc = 0, 0
    model.eval()
    with torch.inference_mode():
      for X, y in tqdm(data_loader):

        X, y = X.to(device), y.to(device)

        y_pred = model(X)
        loss += loss_fn(y_pred, y)
        acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))

      loss /= len(data_loader)
      acc /= len(data_loader)
    return {"model_name": model.__class__.__name__, # only works when model was created with a class
            "model_loss": loss.item(),
            "model_acc": acc}



def download_food101(destination="data/food101"):
    url = "http://data.vision.ee.ethz.ch/cvl/food-101.tar.gz"
    dest_path = Path(destination)
    dest_path.mkdir(parents=True, exist_ok=True)

    tar_path = dest_path / "food-101.tar.gz"

    if not tar_path.exists():
        print("Downloading Food-101 dataset...")
        r = requests.get(url, stream=True)
        with open(tar_path, 'wb') as f:
            f.write(r.content)

    print("Extracting...")
    import tarfile
    with tarfile.open(tar_path, 'r:gz') as tar_ref:
        tar_ref.extractall(path=dest_path)
    print("Done.")


def prepare_food101_subset(target_dir: str = "data/food101_selected", classes=None):
    import torchvision
    from torchvision.datasets import Food101
    from pathlib import Path
    import shutil

    if classes is None:
        classes = ["pizza", "steak", "sushi"]

    root_path = Path("data")
    images_root = root_path / "food-101" / "images"
    output_train = Path(target_dir) / "train"
    output_test = Path(target_dir) / "test"

    # Download Food101 if not already present
    print("Downloading Food101 (if needed)...")
    Food101(root=root_path, split="train", download=True)
    Food101(root=root_path, split="test", download=True)

    print("Preparing selected Food101 subset...")
    # Load splits
    train_data = Food101(root=root_path, split="train", download=False)
    test_data = Food101(root=root_path, split="test", download=False)

    # Internal helper to copy images
    def copy_images(data, split_dir):
        for img_path, label in zip(data._image_files, data._labels):
            if label in classes:
                class_dir = split_dir / label
                class_dir.mkdir(parents=True, exist_ok=True)
                src = images_root / img_path
                dst = class_dir / src.name
                if not dst.exists():
                    shutil.copy(src, dst)

    copy_images(train_data, output_train)
    copy_images(test_data, output_test)

    print(f"Subset ready at: {Path(target_dir).absolute()}")
    return Path(target_dir)


def get_food101_subset(destination: str = "data/pizza_steak_sushi", classes: list = None) -> Path:
    """
    Prepares a subset of Food-101 from already-extracted data.
    Expects the full dataset to be in: data/food-101/images/
    """
    base_path = Path(__file__).parent.parent.parent / "data"
    dest_path = Path(destination)

    if not base_path.exists():
        raise FileNotFoundError(
            f"Expected Food-101 images at {base_path}. Please extract the full dataset there."
        )

    if dest_path.exists():
        print(f"{dest_path} already exists. Skipping subset creation.")
        return dest_path

    if classes is None:
        classes = ["pizza", "steak", "sushi"]

    import shutil
    import random
    dest_path.mkdir(parents=True, exist_ok=True)
    for split in ["train", "test"]:
        for cls in classes:
            src = base_path / cls
            dst = dest_path / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            all_images = list(src.glob("*.jpg"))
            random.shuffle(all_images)
            n_total = 1000 if split == "train" else 250
            for img in all_images[:n_total]:
                shutil.copy(img, dst)

    print(f"Subset created at {dest_path}")
    return dest_path