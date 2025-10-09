import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
  layers.Dense(64, activation='relu', input_shape=(32,)),
  layers.Dense(64, activation='relu'),
  layers.Dense(10, activation='softmax')
])

model.summary()

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(tf.random.normal((1000, 32)), 
          tf.random.uniform((1000,), 
                            maxval=10, 
                            dtype=tf.int64), 
          epochs=10)

model.predict(tf.random.normal((1, 32)))
