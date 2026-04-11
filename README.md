# Federated Learning with MNIST

## Overview
This project implements a **Federated Learning** system using the MNIST handwritten digit dataset. Federated Learning enables multiple clients to collaboratively train a machine learning model without sharing their raw data with a central server.

## What is Federated Learning?
Federated Learning is a distributed machine learning approach where:
- Multiple clients (devices) train the model locally on their data
- Only model weights/parameters are sent to a central server
- The server aggregates the updates from all clients
- The aggregated model is sent back to clients for the next training round
- This process repeats until convergence

**Benefits:**
- ✅ Data Privacy: Raw data never leaves the client
- ✅ Reduced Bandwidth: Only model parameters are transmitted
- ✅ Distributed Learning: Leverage computational power across multiple devices

## Project Structure
```
FEDERATED-LEARNING/
├── model.py          # Neural network model definition
├── dataset.py        # MNIST dataset loading and non-IID distribution
├── client.py         # Federated client implementation
└── server.py         # Federated server (aggregator) implementation
```

## Components

### 1. **model.py**
Defines a simple feedforward neural network for MNIST classification:
- Input Layer: 28×28 = 784 pixels (flattened)
- Hidden Layer: 128 neurons with ReLU activation
- Output Layer: 10 neurons (digits 0-9)

### 2. **dataset.py**
Handles MNIST data loading and distribution:
- Downloads MNIST dataset (60,000 training + 10,000 test images)
- Implements **Non-IID (Non-Independent and Identically Distributed)** data split
- Distributes data across multiple clients with unequal data distributions
- Returns PyTorch DataLoaders for efficient batch processing

### 3. **client.py**
Implements the Federated Client using Flower framework:
- Receives model parameters from the server
- Trains locally on assigned data for 10 epochs
- Calculates performance metrics: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- Sends updated parameters back to the server

## Key Features

### Non-IID Data Distribution
The dataset is split among clients in a non-IID manner, simulating real-world scenarios where:
- Each client has different data distribution
- Clients may have different digit preferences
- Data is not uniformly distributed across clients

### Performance Metrics
For each client, we track:
- **Accuracy**: Percentage of correctly classified digits
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Detailed classification breakdown

### Federated Aggregation
- Uses Flower FL framework for client-server communication
- SGD optimizer with learning rate = 0.01
- Centralized server aggregates model updates
- Communication-efficient parameter sharing

## Requirements
```
torch
torchvision
flwr (Flower)
numpy
scikit-learn
```

## Installation
```bash
pip install torch torchvision flwr numpy scikit-learn
```

## Usage

### Start the Federated Server
```bash
python server.py
```

### Start Federated Clients (in separate terminals)
```bash
# Client 0
python client.py 0

# Client 1
python client.py 1

# Client 2
python client.py 2
```

The server will coordinate training rounds, aggregate model parameters, and evaluate performance.

## Results
After federated training:
- All clients contribute to a single global model
- Individual client metrics show per-client performance
- Aggregated model achieves competitive accuracy on MNIST

## Technology Stack
- **Framework**: PyTorch (Deep Learning)
- **Federated Learning**: Flower (FL)
- **Dataset**: MNIST
- **Optimization**: SGD
- **Metrics**: scikit-learn

## Learning Outcomes
This project demonstrates:
1. How federated learning preserves privacy while enabling collaborative learning
2. Implementation of federated client-server architecture
3. Handling non-IID data distributions in distributed settings
4. Model aggregation and parameter sharing
5. Performance evaluation in distributed environments

## Author
ARAVINDH-1505

## License
MIT

---
**Note**: This is an educational implementation to understand federated learning concepts using MNIST. For production systems, consider privacy-preserving techniques like differential privacy and secure aggregation.