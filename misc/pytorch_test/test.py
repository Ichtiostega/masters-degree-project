from torch import nn
import torch.nn.functional as F
import torch
import torchvision
from pprint import pprint
import matplotlib.pyplot as plt
import toml
from torch import optim

conf = toml.load("config.toml")
train_c = conf["training"]
model_c = conf["model"]

torch.backends.cudnn.enabled = False
torch.manual_seed(1)


def data_loader(batch_size_train: int, batch_size_test: int):
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "files/",
            train=True,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_train,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "files/",
            train=False,
            download=True,
            transform=torchvision.transforms.Compose(
                [
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize((0.1307,), (0.3081,)),
                ]
            ),
        ),
        batch_size=batch_size_test,
        shuffle=True,
    )
    return train_loader, test_loader


def visualize_data(data_loader: torch.utils.data.DataLoader):
    examples = enumerate(data_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    print(example_data.shape)

    plt.figure()
    for i in range(6):
        plt.subplot(2, 3, i + 1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    plt.show()


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

    def fit(
        self, train_loader: torch.utils.data.DataLoader, optimizer, n_epochs
    ):
        self.train()
        for epoch in range(1, n_epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = self(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % train_c["log_interval"] == 0:
                    print(
                        "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                            epoch,
                            batch_idx * len(data),
                            len(train_loader.dataset),
                            100.0 * batch_idx / len(train_loader),
                            loss.item(),
                        )
                    )

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
        print(
            "\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )


def train_cycle(
    network: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    epochs: int,
):
    network.test(test_loader)
    network.fit(train_loader, optimizer, epochs)
    network.test(test_loader)


def main():
    train_loader, test_loader = data_loader(
        train_c["batch_size_train"], train_c["batch_size_test"]
    )
    net = Net()
    optimizer = optim.SGD(
        net.parameters(), lr=train_c["learning_rate"], momentum=train_c["momentum"]
    )
    # visualize_data(test_loader)
    train_cycle(net, train_loader, test_loader, optimizer, train_c["n_epochs"])


if __name__ == "__main__":
    main()
