import torch
import time
from torch import nn
from tqdm.auto import tqdm
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torchinfo import summary


print(f"Torch version: {torch.__version__}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Setting device to: {device}")


train_data = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)


test_data = datasets.MNIST(
    root = "data",
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor(),
    target_transform=None
)

BATCH_SIZE = 32
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True) # shuffle = True -> reshuffles the data at every epoch to avoid overfitting
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)



class MNISTModel1(nn.Module):
    def __init__(self, in_features: int, hidden_units: int, out_features: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_features,
                      out_channels=hidden_units,
                      kernel_size=5,
                      stride=1,
                      padding=1
                     ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                        stride=2),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1
                     ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2
                        )
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units*6*6,
                      out_features=out_features
                     )
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block_1(x)
        return self.classifier(x)
    


model_1 = MNISTModel1(in_features=1,
                     hidden_units=64,
                     out_features=10).to(device)


print(summary(model_1, input_size=(1, 1, 28, 28))) # seeing a single MNIST digit going through the model to see the parameters

loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model_1.parameters(),
    lr = 0.1
)

print(f"Model set-up: {next(model_1.parameters()).shape}")


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
               accuracy_fn,
               device: torch.device = device):
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




def train_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn,
              accuracy_fn,
              optimizer: torch.optim.Optimizer,
              device: torch.device = device):

    train_loss, train_acc = 0, 0

    model.to(device)
    model.train()

    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)

        y_logits = model(X)

        loss = loss_fn(y_logits, y)
        train_loss+=loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_logits.argmax(dim=1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Training loss: {train_loss} | Training accuracy: {train_acc}")

def test_step(model: torch.nn.Module,
              data_loader: torch.utils.data.DataLoader,
              loss_fn,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.inference_mode():
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)

            test_pred = model(X)

            test_loss += loss_fn(test_pred, y)
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))

        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")
    



epochs = 30

start_time = time.time()

for epoch in range(epochs):
    train_step(model=model_1,
               data_loader=train_dataloader,
               loss_fn=loss_fn,
               accuracy_fn=accuracy_fn,
               optimizer=optimizer,
               device=device
              )

    if (epoch % 3) == 0:
        test_step(model=model_1,
                 data_loader=test_dataloader,
                 loss_fn=loss_fn,
                 accuracy_fn=accuracy_fn,
                 device=device)

end_time = time.time()
training_time = end_time - start_time
print(f"Total training time: {training_time:.2f} seconds")

model_1_results = eval_model(
    model=model_1,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn = accuracy_fn,
    device=device
)

print(f"Model results: {model_1_results}")
    
        