"""
Car Classifier Inference Script
--------------------------------
Reloads a trained PyTorch model and predicts class labels for new car data.
"""

import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

# -------------------------
# 1. Define the same model class
# -------------------------
class CarClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CarClassifier, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.network(x)

# -------------------------
# 2. Recreate the model and load weights
# -------------------------
model = CarClassifier(input_dim=6, hidden_dim=16, num_classes=4)
model.load_state_dict(torch.load("car_model.pt"))
model.eval()
print("Model loaded successfully!\n")

# -------------------------
# 3. Define example car data
# -------------------------
# Features: buying, maint, doors, persons, lug_boot, safety
# These must be encoded exactly as during training
example_cars = np.array([
    [0, 0, 2, 2, 0, 2],  # example car 1
    [3, 2, 1, 1, 1, 0],  # example car 2
    [2, 1, 0, 1, 2, 1],  # example car 3
], dtype=np.float32)

car_tensor = torch.tensor(example_cars)

# -------------------------
# 4. Make predictions
# -------------------------
with torch.no_grad():
    outputs = model(car_tensor)
    _, predicted_classes = torch.max(outputs, 1)

# -------------------------
# 5. Decode predicted labels
# -------------------------
# Define the same label mapping as training
class_names = np.array(['unacc', 'acc', 'good', 'vgood'])
predicted_labels = class_names[predicted_classes.numpy()]

# Print results
for i, label in enumerate(predicted_labels):
    print(f"Car {i+1} predicted class: {label}")
