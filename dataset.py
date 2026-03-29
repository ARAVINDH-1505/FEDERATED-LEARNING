import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import DataLoader, TensorDataset

def load_data(num_clients=3):
    # Download MNIST
    train_dataset = datasets.MNIST(
        root=r"D:\VELAI THEDUM PADALAM\FEDERATED LEARNING\mnist_data",
        train=True,
        download=True
    )

    test_dataset = datasets.MNIST(
        root=r"D:\VELAI THEDUM PADALAM\FEDERATED LEARNING\mnist_data",
        train=False,
        download=True
    )

    # Convert to numpy
    X = train_dataset.data.numpy() / 255.0
    y = train_dataset.targets.numpy()

    # NON-IID Split (IMPORTANT)
    idx = np.argsort(y)
    X, y = X[idx], y[idx]

    shard_size = len(X) // num_clients
    clients = []

    for i in range(num_clients):
        start = i * shard_size
        end = start + shard_size
        clients.append((X[start:end], y[start:end]))

    # Convert to DataLoader
    trainloaders = []
    for X_client, y_client in clients:
        X_tensor = torch.tensor(X_client, dtype=torch.float32).unsqueeze(1)
        y_tensor = torch.tensor(y_client, dtype=torch.long)

        dataset = TensorDataset(X_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        trainloaders.append(loader)

    # Test loader
    X_test = test_dataset.data.numpy() / 255.0
    y_test = test_dataset.targets.numpy()

    X_test = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1)
    y_test = torch.tensor(y_test, dtype=torch.long)

    testloader = DataLoader(TensorDataset(X_test, y_test), batch_size=32)

    return trainloaders, testloader