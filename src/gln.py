import sys
from torchvision.models import googlenet
from torch import optim
import torchvision
import torch
import logging
import toml

conf = toml.load("config.toml")
train_c = conf["training"]
model_c = conf["model"]

torch.backends.cudnn.enabled = False
torch.manual_seed(1)

logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
logging.getLogger().level = logging.INFO


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


def fit(net, train_loader: torch.utils.data.DataLoader, optimizer, n_epochs):
    net.train()
    logging.info(f"{type(net).__name__} training.")
    for epoch in range(1, n_epochs + 1):
        for batch_idx, (data, target) in enumerate(train_loader):
            optimizer.zero_grad()
            output = net(data)
            loss = torch.nn.NLLLoss()(output.logits, target)
            loss.backward()
            optimizer.step()
            if batch_idx % train_c["log_interval"] == 0:
                logging.info(
                    "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                        epoch,
                        batch_idx * len(data),
                        len(train_loader.dataset),
                        100.0 * batch_idx / len(train_loader),
                        loss.item(),
                    )
                )


model = googlenet(num_classes=101)
food_train, food_test, flower_train, flower_test = data_loader(
    train_c["batch_size_train"], train_c["batch_size_test"]
)
optimizer = optim.SGD(
    model.parameters(), lr=train_c["learning_rate"], momentum=train_c["momentum"]
)
fit(model, food_train, optimizer, 3)
