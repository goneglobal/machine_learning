# Install dependencies (if running locally)
# pip install torch torchvision pandas scikit-learn matplotlib seaborn

"""
Machine Learning (CSE5ML) Example
Supervised Learning — Car Evaluation Classification using PyTorch
---------------------------------------------------------------
Includes: training, accuracy, precision/recall/F1, confusion matrix heatmap
"""

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. Load and inspect data
# -------------------------
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data"
columns = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'class']
df = pd.read_csv(url, names=columns)

print("Sample data:")
print(df.head(), "\n")

# -------------------------
# 2. Encode categorical data
# -------------------------
label_encoders = {}
for col in df.columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('class', axis=1).values
y = df['class'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# 3. PyTorch Dataset
# -------------------------
class CarDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_data = CarDataset(X_train, y_train)
test_data = CarDataset(X_test, y_test)
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32)

# -------------------------
# 4. Define Model
# -------------------------
class CarClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(CarClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        return self.network(x)

model = CarClassifier(input_dim=6, hidden_dim=16, num_classes=4)

# -------------------------
# 5. Train Model
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
epochs = 30
loss_values = []

print("Training started...\n")
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    loss_values.append(total_loss / len(train_loader))
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss_values[-1]:.4f}")

print("\nTraining complete.\n")

# Save the trained model parameters
torch.save(model.state_dict(), "car_model.pt")
print("Model saved as 'car_model.pt'")

# -------------------------
# 6. Evaluate Model
# -------------------------
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)
        _, preds = torch.max(outputs, 1)
        y_true.extend(y_batch.numpy())
        y_pred.extend(preds.numpy())

acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)
class_names = label_encoders['class'].inverse_transform([0, 1, 2, 3])

print(f"Test Accuracy: {acc:.3f}\n")
print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

# -------------------------
# 7. Confusion Matrix Visualization
# -------------------------
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix - Car Classifier")
plt.tight_layout()
plt.show()

# -------------------------
# 8. Loss Curve Visualization
# -------------------------
plt.figure(figsize=(6,4))
plt.plot(range(1, epochs+1), loss_values, marker='o')
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(True)
plt.tight_layout()
plt.show()
