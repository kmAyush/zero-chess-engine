#!/usr/bin/env python3
from state import State
from train import NeuralNet
import torch

if __name__ == "__main__":
    model = torch.load('models/examples_1k.pth')

    data = State().serialize()[None]
    output = model(torch.tensor(data).float())
    print(output)