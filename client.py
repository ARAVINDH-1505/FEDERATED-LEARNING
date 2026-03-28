import flwr as fl
import torch
from model import Net
from dataset import load_data
import sys
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import numpy as np

cid = sys.argv[1] 

DEVICE = torch.device("cpu")

trainloaders, testloader = load_data()

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, cid):
        self.model = Net().to(DEVICE)
        self.trainloader = trainloaders[int(cid)]

    def get_parameters(self, config):
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters):
        keys = self.model.state_dict().keys()
        state_dict = dict(zip(keys, parameters))
        self.model.load_state_dict({k: torch.tensor(v) for k, v in state_dict.items()})

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.01)

        self.model.train()
        for epoch in range(10):
            for input1, input2 in self.trainloader:
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(self.model(input1), input2)
                loss.backward()
                optimizer.step()

        return self.get_parameters(config), len(self.trainloader.dataset), {}

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for input1, input2 in testloader:
                input1, input2 = input1.to(DEVICE), input2.to(DEVICE)
                outputs = self.model(input1)

                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(input2.cpu().numpy())

        # Convert to numpy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        # Metrics
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average="weighted", zero_division=0)
        recall = recall_score(all_labels, all_preds, average="weighted", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)

        cm = confusion_matrix(all_labels, all_preds)

        print(f"\nClient {self.trainloader}:")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("Confusion Matrix:\n", cm)

        loss = 0.0  # optional

        return loss, len(testloader.dataset), {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }


def client_fn(cid):
    return FlowerClient(cid)

if __name__ == "__main__":
    fl.client.start_numpy_client(
        server_address="localhost:8080",
        client=client_fn(cid)
    )