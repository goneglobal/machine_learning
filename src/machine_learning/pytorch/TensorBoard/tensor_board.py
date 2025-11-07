## how to use htis
# $ python tensor_board.py 
# tensorboard --logdir=runs
# http://localhost:6006/

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid

# Define a multi-layer neural network
class MultiLayerNet(nn.Module):
    def __init__(self, input_size=100, hidden_sizes=[128, 64, 32], output_size=10):
        super(MultiLayerNet, self).__init__()
        layers = []
        last_size = input_size
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(last_size, hidden_size))
            layers.append(nn.ReLU())
            last_size = hidden_size
        layers.append(nn.Linear(last_size, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# Hyperparameters
input_size = 100
output_size = 10
batch_size = 32
num_epochs = 5

# Create dummy dataset (random inputs and targets)
data = torch.randn(500, input_size)
targets = torch.randint(0, output_size, (500,))

# Instantiate model, loss, optimizer
model = MultiLayerNet(input_size=input_size, output_size=output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Setup TensorBoard SummaryWriter
writer = SummaryWriter(log_dir="runs/multilayer_example")

# Add model graph to TensorBoard (using a dummy input)
dummy_input = torch.randn(batch_size, input_size)
writer.add_graph(model, dummy_input)

# Training loop
for epoch in range(num_epochs):
    running_loss = 0.0
    correct = 0

    for i in range(0, len(data), batch_size):
        inputs = data[i:i+batch_size]
        labels = targets[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / (len(data) / batch_size)
    epoch_accuracy = correct / len(data)

    # Log scalar values to TensorBoard
    writer.add_scalar('Loss/train', epoch_loss, epoch)
    writer.add_scalar('Accuracy/train', epoch_accuracy, epoch)

    print(f"Epoch {epoch+1} - Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

# Close the writer
writer.close()
