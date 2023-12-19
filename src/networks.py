import torch as torch
import torch.nn as nn
import torch.nn.functional as F

# a CNN network that takes in the board representation s and outputs a tuple of 2 values, p and v
# s is a 10*10*2 tensor of the board state,
# p is a 100*1 tensor of policy probabilities for each move, v is a scalar value of the win rate
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        # input is 10*10*2
        # residual tower: a convolutional layer followed by 10 residual blocks
        # module 1: convolutional layer with (1) 64 filters of kernel size 3*3 with stride 1, (2)bn, (3)relu
        # module 2: 10 residual blocks with (1) 64 filters, (2)bn, (3)relu, (4) 64 filters, (5)bn, (6)skip connection, (7)relu
        # output: two heads, one for policy and one for value
        # policy head: convolutional layer with (1) 2 filters, (2)bn, (3)relu, (4)fully connected layer with 10*10 nodes, (5)softmax
        # value head: convolutional layer with (1) 1 filter, (2)bn, (3)relu, (4)fully connected layer with 64 nodes, (5)relu, (6)fully connected layer with 1 node, (7)tanh
        self.conv = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.residual = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )
        self.policy = nn.Sequential(
            nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(200, 100),
            nn.Softmax(dim=1)
        )
        self.value = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh()
        )
    
    def forward(self, x) -> tuple[torch.Tensor, torch.Tensor]:
        # print(x.shape)
        x = self.conv(x)
        for i in range(10):
            # print(x.shape, self.residual(x).shape)
            x = self.residual(x) + x
        # print(x.shape)
        # print(f"Memory size of tensor x: {x.element_size() * x.numel()} bytes")
        p = self.policy(x)
        # print(p.shape)
        # print(x.shape)
        v = self.value(x).squeeze()
        # print(v.shape)
        return p, v
