# Release Year Prediction
## Estimate release_year from the description text and cast — this is noisy but can test temporal signal detection.

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer

# Pull Data & Display Head
df = pd.read_csv("./netflix_titles.csv")
print(df.head(5))

# Features - show_id,type,title,director,cast,country,date_added,release_year,rating,duration,listed_in,description
# df['text'] = df['type'].fillna("") + ' ' + df['title'].fillna("") + ' ' + df['cast'].fillna("")
# df['features'] = df['director'].fillna("") + ' ' + df['cast'].fillna("")
df['features'] = (
    df['director'].fillna('') + ' ' +
    df['cast'].fillna('') + ' ' +
    df['listed_in'].fillna('') + ' ' +
    df['type'].fillna('') + ' ' +
    df['title'].fillna('') + ' ' +
    df['description'].fillna('')
)

# Drop Rows with out 'release_year' value
df = df.dropna(subset='release_year')

# Drop Years with only 1 Sample in Data
count = df['release_year'].value_counts()
df = df[df['release_year'].isin(count[count > 1].index)]

# Encode Labels
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['release_year'])
num_classes = len(label_encoder.classes_)
print("Classes:", label_encoder.classes_)

# Create Test/Train Data Sets
X_train, X_test, y_train, y_test = train_test_split(
  df['features'], df['label'],
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

# Tokenization (required for Neural Networks)
MAX_VOCAB = 20000   # 20,000 more frequent words
MAX_LEN = 200       # 200 tokens is expexted length (required)
tokenizer = Tokenizer(num_words=MAX_VOCAB, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

# Padding (required for Neural Networks)
X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(
  tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN, padding='post', truncating='post'
)
X_test_pad  = tf.keras.preprocessing.sequence.pad_sequences(
  tokenizer.texts_to_sequences(X_test),  maxlen=MAX_LEN, padding='post', truncating='post'
)

# Build model
EMBED_DIM = 128
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(MAX_VOCAB, EMBED_DIM),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
    tf.keras.layers.GlobalMaxPooling1D(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    # Clasification Model
    # tf.keras.layers.Dense(num_classes, activation='softmax')
    # Regresion Model
    tf.keras.layers.Dense(1)
])
# Clasification Model
# model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# Regresion Model
model.compile(loss='mse', optimizer='adam', metrics=['mae'])
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