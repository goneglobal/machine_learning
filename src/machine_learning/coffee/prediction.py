import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Load data
df = pd.read_csv("coffee_sales.csv")

# Features (inputs)
feature_cols = ["hour_of_day", "cash_type", "Time_of_Day", "Weekday", "Month_name"]

# Target (coffee choice)
target_col = "coffee_name"

# Encode target correctly
target_lookup = layers.StringLookup(output_mode="int", vocabulary=df[target_col].unique())
df["coffee_label"] = target_lookup(df[target_col].values).numpy() - 1

# Number of classes
num_classes = len(target_lookup.get_vocabulary())

# Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[feature_cols], df["coffee_label"], test_size=0.2, random_state=42)

# Preprocessing layers
inputs = {}
encoded_features = []

# Categorical columns
cat_cols = ["cash_type", "Time_of_Day", "Weekday", "Month_name"]
for col in cat_cols:
    inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.string)
    lookup = layers.StringLookup(output_mode="one_hot")
    lookup.adapt(df[col].unique())
    encoded = lookup(inp)
    inputs[col] = inp
    encoded_features.append(encoded)

# Numeric columns
num_cols = ["hour_of_day"]
for col in num_cols:
    inp = tf.keras.Input(shape=(1,), name=col, dtype=tf.float32)
    norm = layers.Normalization()
    norm.adapt(df[col].values.reshape(-1,1))
    encoded = norm(inp)
    inputs[col] = inp
    encoded_features.append(encoded)

# Concatenate all
all_features = layers.concatenate(encoded_features)

# Build classifier
x = layers.Dense(64, activation="relu")(all_features)
x = layers.Dropout(0.3)(x)
x = layers.Dense(32, activation="relu")(x)
output = layers.Dense(len(df[target_col].unique()), activation="softmax")(x)

model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(optimizer="adam", 
              loss="sparse_categorical_crossentropy", 
              metrics=["accuracy"])

model.summary()


# Convert training data into dict for Keras
train_dict = {col: X_train[col].values for col in feature_cols}
test_dict  = {col: X_test[col].values  for col in feature_cols}

history = model.fit(train_dict, y_train, 
                    validation_data=(test_dict, y_test),
                    epochs=25, batch_size=32)



# Predict probabilities
pred_probs = model.predict(test_dict)

# Convert probabilities -> class indices
pred_labels = np.argmax(pred_probs, axis=1)

# Map indices back to coffee names
vocab = target_lookup.get_vocabulary()  # same lookup used for training
vocab = vocab[1:]  # drop [UNK], since we subtracted 1 earlier

coffee_preds = [vocab[i] for i in pred_labels]

print("Sample predictions:")
print(coffee_preds[:10])

for i in range(5):  # show first 5 samples
    probs = pred_probs[i]
    top3 = np.argsort(probs)[-3:][::-1]  # top 3 predictions
    print(f"Sample {i}:")
    for idx in top3:
        print(f"  {vocab[idx]} -> {probs[idx]:.2f}")