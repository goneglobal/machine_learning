# 📦 1. Install Dependencies
# pip install pandas scikit-learn tensorflow

# 📂 2. Load and Prepare Data
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load your CSV
df = pd.read_csv("netflix_titles.csv")

# Combine description and cast into a single text field
df['text'] = df['description'].fillna('') + ' ' + df['cast'].fillna('')

# Drop rows with missing rating
df = df.dropna(subset=['rating'])

# Encode labels (ratings)
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['rating'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)

# 📝 3. Text Tokenization & Padding
# We’ll convert text to integer sequences for the neural net.

MAX_VOCAB = 20000       # top words to keep
MAX_LEN = 200           # max tokens per sample

tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq  = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad  = pad_sequences(X_test_seq,  maxlen=MAX_LEN, padding='post', truncating='post')

# 🔥 4. Build the TensorFlow Model
# We’ll use an Embedding → Bi-LSTM → Dense classifier.

EMBED_DIM = 128

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=MAX_VOCAB, output_dim=EMBED_DIM, input_length=MAX_LEN),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# 🚀 5. Train the Model
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=5,
    batch_size=32,
    verbose=1
)

# 📈 6. Evaluate
loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# 🔮 7. Make Predictions on New Data
# def predict_rating(description, cast):
#     text = description + " " + cast
#     seq = tokenizer.texts_to_sequences([text])
#     pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
#     pred = model.predict(pad)
#     rating_idx = np.argmax(pred)
#     return label_encoder.inverse_transform([rating_idx])[0]

# example_desc = "A group of kids discover a haunted house in their small town."
# example_cast = "John Doe, Jane Smith"

# print("Predicted rating:", predict_rating(example_desc, example_cast))