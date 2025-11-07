import torch
import torch.nn as nn
import torch.optim as optim

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)  # input 10 features, output 20
        self.act1 = nn.LeakyReLU(negative_slope=0.01)
        self.fc2 = nn.Linear(20, 5)   # output layer with 5 outputs
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)   # LeakyReLU activation
        x = self.fc2(x)
        return x

# Create model instance
model = SimpleNet()

# Example input: batch of 3 samples, each with 10 features
inputs = torch.randn(3, 10)

# Forward pass
outputs = model(inputs)

print("Input:", inputs)
print("Output:", outputs)
