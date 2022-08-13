import os
from pathlib import Path
import sys
from torch import nn
import torch.nn.functional as F
import torch
import torchvision
import matplotlib.pyplot as plt
import toml
from torch import optim
import logging

logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.getLogger().addHandler(logging.FileHandler("execution.log"))
logging.getLogger().level = logging.INFO

conf = toml.load("config.toml")
train_c = conf["training"]
model_c = conf["model"]

torch.backends.cudnn.enabled = False
torch.manual_seed(1)


def data_loader(batch_size_train: int, batch_size_test: int):
    food_train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.Food101(
            root="data",
            split="train",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    food_test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.Food101(
            root="data",
            split="test",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )

    flower_train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.Flowers102(
            root="data",
            split="train",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    flower_test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.Flowers102(
            root="data",
            split="test",
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.RandomResizedCrop(224),
                    torchvision.transforms.ToTensor(),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )
    return food_train_loader, food_test_loader, flower_train_loader, flower_test_loader


def visualize_data(data_loader: torch.utils.data.DataLoader):
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    logging.info(example_data.shape)

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(
            torchvision.transforms.ToPILImage()(example_data[i]), interpolation="none"
        )
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


class AugmentedNet(nn.Module):
    test_acc: list[float] = []
    train_loss: list[float] = []
    test_loss: list[float] = []
    train_counter: list[int] = []

    def save(self, dir: Path):
        os.makedirs(dir, exist_ok=True)
        data = {
            "test_acc": self.test_acc,
            "train_loss": self.train_loss,
            "test_loss": self.test_loss,
            "train_counter": self.train_counter,
            "state_dict": self.state_dict(),
        }
        torch.save(data, dir / f"{type(self).__name__}.msd")

    def load(self, dir: Path):
        data = torch.load(dir / f"{type(self).__name__}.msd")
        self.test_acc = data["test_acc"]
        self.train_loss = data["train_loss"]
        self.test_loss = data["test_loss"]
        self.train_counter = data["train_counter"]
        self.load_state_dict(data["state_dict"])

    def fit(self, train_loader: torch.utils.data.DataLoader, optimizer, n_epochs):
        self.train()
        logging.info(f"{type(self).__name__} training.")
        for epoch in range(1, n_epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                loss = nn.NLLLoss()(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % train_c["log_interval"] == 0:
                    self.train_loss.append(loss.item())
                    self.train_counter.append(
                        (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset))
                    )
                    logging.info(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )
        self.save(Path("models"))

    def test(self, test_loader: torch.utils.data.DataLoader):
        self.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                output = self(data)
                test_loss += F.nll_loss(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        self.test_loss.append(test_loss)
        acc = correct / len(test_loader.dataset)
        self.test_acc.append(acc)
        logging.info(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * acc,
            )
        )


class FoodNet(AugmentedNet):
    def __init__(self):
        super(FoodNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 1, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.max_pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(529, 300)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(300, 101)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.max_pool2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


class FlowerNet(AugmentedNet):
    def __init__(self):
        super(FlowerNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=5)
        self.max_pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(3, 1, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.max_pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(529, 300)
        self.dropout = nn.Dropout()
        self.fc2 = nn.Linear(300, 102)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.max_pool1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.conv2_drop(x)
        x = self.max_pool2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return self.softmax(x)


def fit_wrapper(
    net: nn.Module, train_loader: torch.utils.data.DataLoader, optimizer, n_epochs
):
    """Fits if no trained network exists."""
    if f"{type(net).__name__}.msd" in os.listdir("models"):
        net.load(Path("models"))
    else:
        net.fit(train_loader, optimizer, n_epochs)
    return net


def main():
    (
        food_train_loader,
        food_test_loader,
        flower_train_loader,
        flower_test_loader,
    ) = data_loader(train_c["batch_size_train"], train_c["batch_size_test"])

    net = FlowerNet()
    optimizer = optim.SGD(
        net.parameters(), lr=train_c["learning_rate"], momentum=train_c["momentum"]
    )
    fit_wrapper(net, flower_train_loader, optimizer, train_c["n_epochs"])
    net = FoodNet()
    print("UNTRAINED")
    for param in net.parameters():
        print(param.data)
    optimizer = optim.SGD(
        net.parameters(), lr=train_c["learning_rate"], momentum=train_c["momentum"]
    )
    fit_wrapper(net, food_train_loader, optimizer, train_c["n_epochs"])
    print("TRAINED")
    for param in net.parameters():
        print(param.data)


if __name__ == "__main__":
    main()
