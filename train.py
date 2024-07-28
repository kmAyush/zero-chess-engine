#!/usr/bin/env python
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from torch import optim

class ValueDataset(Dataset):
    def __init__(self, data_size):
        data = np.load(f"serialized_data/dataset_{data_size}.npz")
        self.X = data['arr_0']
        self.Y = data['arr_1']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.Y[index])

class NeuralNet(nn.Module):
    def __init__(self):
        super(NeuralNet, self).__init__()
        input_channel = 5
        copy = [16, 32, 64, 128]
        
        self.a1 = nn.Conv2d(input_channel, copy[0], kernel_size = 3, padding=1)
        # for B=32
        # [32, 5, 8, 8]  => [32, 16, 8, 8] 

        self.a2 = nn.Conv2d(copy[0], copy[0], kernel_size = 3, padding = 1)
        # [32, 16, 8, 8] => [32, 16, 8, 8]

        self.a3 = nn.Conv2d(copy[0], copy[1], kernel_size = 3, stride = 2)
        # [32, 16, 8, 8] => [32, 32, 3, 3]

        
        self.b1 = nn.Conv2d(copy[1], copy[1], kernel_size = 3, padding = 1)
        # [32, 32, 3, 3] => [32, 32, 3, 3]

        self.b2 = nn.Conv2d(copy[1], copy[1], kernel_size = 3, padding = 1)
        # [32, 32, 3, 3] => [32, 32, 3, 3]

        self.b3 = nn.Conv2d(copy[1], copy[2], kernel_size = 3, stride = 2)
        # [32, 32, 3, 3] => [32, 64, 1, 1]


        self.c1 = nn.Conv2d(copy[2], copy[2], kernel_size = 2, padding = 1)
        # [32, 64, 1, 1] => [32, 64, 2, 2] 

        self.c2 = nn.Conv2d(copy[2], copy[2], kernel_size = 2, padding = 1)
        # [32, 64, 2, 2] => [32, 64, 3, 3]

        self.c3 = nn.Conv2d(copy[2], copy[3], kernel_size = 2, stride = 2)
        # [32, 64, 3, 3] => [32, 128, 1, 1]
        
    
        self.d1 = nn.Conv2d(copy[3], copy[3], kernel_size = 1)
        # [32, 128, 1, 1] => [32, 128, 1, 1]

        self.d2 = nn.Conv2d(copy[3], copy[3], kernel_size = 1)
        # [32, 128, 1, 1] => [32, 128, 1, 1]
        
        self.d3 = nn.Conv2d(copy[3], copy[3], kernel_size = 1)
        # [32, 128, 1, 1] => [32, 128, 1, 1]

        self.linear_layer = nn.Linear(128, 1)
        # [32, 128] => [32, 1]

    def forward(self, x):

        x = F.relu(self.a1(x))
        x = F.relu(self.a2(x))
        x = F.relu(self.a3(x))

        # 4x4
        x = F.relu(self.b1(x))
        x = F.relu(self.b2(x))
        x = F.relu(self.b3(x))

        # 2x2
        x = F.relu(self.c1(x))
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))

        
        x = F.relu(self.d1(x))
        x = F.relu(self.d2(x))
        x = F.relu(self.d3(x))

        x = x.view(-1, 128)  # [B, 128, 1, 1] => [B, 128] 
        x = self.linear_layer(x)
        return F.tanh(x)

if __name__ == "__main__":
    data_size = '1M'    
    chess_dataset = ValueDataset(data_size)
    data_loader = DataLoader(chess_dataset, batch_size = 32, shuffle = True)
    model = NeuralNet()
    optimizer = optim.Adam(model.parameters())
    mse_loss = nn.MSELoss()

    device = torch.device("cpu")
    if device == "cuda":
        model.cuda()

    model.train()
    print("Dataset Size : ", len(chess_dataset))

    for epoch in range(100):
        total_loss = 0
        iterations = 0
        for batch_idx, (data, target) in enumerate(data_loader):

            #print("X shape = ", data.shape)
            #print("Y shape = ", target.shape)
            target = target.unsqueeze(-1)
            data, target = data.to(device), target.to(device)
            data = data.float()
            target = target.float()

            optimizer.zero_grad()
            output = model(data)

            loss = mse_loss(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            iterations += 1
        print(f"{epoch} : {total_loss/iterations}")
        torch.save(model, f"models/examples_{data_size}.pth")