import torch
import torch.nn as nn

# Create a LeakyReLU activation layer with default negative slope 0.01
leaky_relu = nn.LeakyReLU()

# Example input tensor (some negative, some positive values)
x = torch.tensor([-3.0, -1.0, 0.0, 2.0, 4.0])

# Apply LeakyReLU
output = leaky_relu(x)

print("Input:", x)
print("Output:", output)
