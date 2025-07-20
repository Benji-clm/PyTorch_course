"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
from pathlib import Path

def get_food101_subset(destination: str = "data/pizza_steak_sushi", classes: list = None) -> Path:
    """
    Prepares a subset of Food-101 from already-extracted data.
    Expects the full dataset to be in: data/food-101/images/
    """
    base_path = Path(__file__).parent.parent.parent.parent / "data"
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
            src = base_path / "food-101" / "images" / cls
            dst = dest_path / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            all_images = list(src.glob("*.jpg"))
            random.shuffle(all_images)
            n_total = 1000 if split == "train" else 250
            for img in all_images[:n_total]:
                shutil.copy(img, dst)

    print(f"Subset created at {dest_path}")
    return dest_path

def save_model(model: torch.nn.Module,
               model_name: str):
  """Saves a PyTorch model to a target directory.

  Args:
    model: A target PyTorch model to save.
    model_name: name of the saved model
  """
  # Create target directory
  MODEL_PATH = Path(__file__).parent / "models"
  MODEL_PATH.mkdir(parents=True,
                    exist_ok=True)
  MODEL_NAME = "Reduced_Food101_resnet"
  MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

  # Save the model state_dict()
  print(f"[INFO] Saving model to: {MODEL_SAVE_PATH}")
  torch.save(obj=model.state_dict(),
             f=MODEL_SAVE_PATH)
