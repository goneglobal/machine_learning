import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Load CSV
df = pd.read_csv("netflix_titles.csv")

# Combine description and cast
df['text'] = df['description'].fillna('') + ' ' + df['cast'].fillna('')

# Drop rows with missing rating
df = df.dropna(subset=['rating'])

# Replace rare classes
threshold = 5
counts = df['rating'].value_counts()
df['rating'] = df['rating'].apply(lambda x: x if counts[x] >= threshold else 'Other')

# Encode labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['rating'])
num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)

# Train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'],
    test_size=0.2,
    random_state=42,
    stratify=df['label']
)

# Class Weights
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights = dict(enumerate(class_weights))

# Tokenization & padding
MAX_VOCAB = 20000
MAX_LEN = 200
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_pad = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad  = pad_sequences(tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding='post', truncating='post')

# Build model
EMBED_DIM = 128
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_VOCAB, EMBED_DIM),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=3, 
    restore_best_weights=True
)
history = model.fit(
    X_train_pad, y_train,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    class_weight=class_weights,
    callbacks=[early_stop]
)

# Evaluate
loss, acc = model.evaluate(X_test_pad, y_test, verbose=0)
print(f"Test Accuracy: {acc*100:.2f}%")

# # Prediction function
# def predict_rating(description, cast):
#     text = description + " " + cast
#     seq = tokenizer.texts_to_sequences([text])
#     pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
#     pred = model.predict(pad)
#     rating_idx = np.argmax(pred)
#     return label_encoder.inverse_transform([rating_idx])[0]

# # Example
# example_desc = "A group of kids discover a haunted house in their small town."
# example_cast = "John Doe, Jane Smith"
# print("Predicted rating:", predict_rating(example_desc, example_cast))
