import os
import tensorflow as tf
# Use the standard Keras imports instead of tensorflow.python.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
import numpy as np
import matplotlib.pyplot as plt

def build_model_sequential(input_shape, num_classes):
    model = Sequential()
    model.add(Flatten(input_shape=input_shape))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(num_classes, activation='softmax'))
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def load_and_preprocess_data():
    """Load and preprocess MNIST dataset"""
    # Load the famous handwritten digits dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    # Normalize pixel values from 0-255 to 0-1
    x_train, x_test = x_train.astype('float32') / 255.0, x_test.astype('float32') / 255.0
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    return (x_train, y_train), (x_test, y_test)

def train_model(model, train_data, test_data, epochs=5):
    """Train the model and return training history"""
    x_train, y_train = train_data
    x_test, y_test = test_data
    
    print("🚀 Starting training...")
    
    # Add some callbacks for better training
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.5)
    ]
    
    history = model.fit(
        x_train, y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def evaluate_model(model, test_data):
    """Evaluate model performance"""
    x_test, y_test = test_data
    
    print("\n📊 Evaluating model...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {test_accuracy:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    return test_loss, test_accuracy

def make_predictions(model, test_data, num_samples=5):
    """Make predictions on sample images"""
    x_test, y_test = test_data
    
    # Get random samples
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    sample_images = x_test[indices]
    sample_labels = y_test[indices]
    
    # Make predictions
    predictions = model.predict(sample_images, verbose=0)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print("\n🔮 Sample Predictions:")
    for i, (true_label, pred_label, confidence) in enumerate(zip(
        sample_labels, predicted_classes, np.max(predictions, axis=1)
    )):
        status = "✅" if true_label == pred_label else "❌"
        print(f"Sample {i+1}: True={true_label}, Predicted={pred_label}, Confidence={confidence:.3f} {status}")
    
    return indices, predicted_classes, predictions

def visualize_predictions(test_data, indices, predicted_classes, predictions):
    """Visualize some predictions"""
    x_test, y_test = test_data
    
    plt.figure(figsize=(15, 3))
    for i, idx in enumerate(indices):
        plt.subplot(1, len(indices), i+1)
        plt.imshow(x_test[idx], cmap='gray')
        plt.title(f'True: {y_test[idx]}\nPred: {predicted_classes[i]}\nConf: {np.max(predictions[i]):.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('src/predictions.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Saved predictions visualization to src/predictions.png")

def save_model(model, filename='src/trained_model.keras'):
    """Save the trained model in modern Keras format"""
    model.save(filename)
    print(f"💾 Model saved to {filename}")

def plot_training_history(history):
    """Plot training history"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'], label='Training Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history.history['loss'], label='Training Loss')
    ax2.plot(history.history['val_loss'], label='Validation Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('src/training_history.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Saved training history to src/training_history.png")

def load_saved_model(filename='src/trained_model.keras'):
    """Load a saved model for inference"""
    try:
        model = tf.keras.models.load_model(filename)
        print(f"✅ Model loaded from {filename}")
        return model
    except FileNotFoundError:
        print(f"❌ Model file {filename} not found")
        return None

def test_custom_prediction(model, custom_image=None):
    """Test the model with a custom image or create a simple test"""
    if custom_image is None:
        # Create a simple test digit (a crude "7")
        custom_image = np.zeros((28, 28))
        custom_image[5:8, 10:20] = 1.0    # horizontal line
        custom_image[8:25, 17:20] = 1.0   # vertical line
        print("🎨 Created a simple test digit (crude '7')")
    
    # Reshape for model input
    test_input = custom_image.reshape(1, 28, 28)
    
    # Make prediction
    prediction = model.predict(test_input, verbose=0)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)
    
    print(f"🔮 Custom prediction: {predicted_class} (confidence: {confidence:.3f})")
    
    # Visualize
    plt.figure(figsize=(8, 3))
    
    plt.subplot(1, 2, 1)
    plt.imshow(custom_image, cmap='gray')
    plt.title('Custom Test Image')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), prediction[0])
    plt.title(f'Prediction: {predicted_class}\nConfidence: {confidence:.3f}')
    plt.xlabel('Digit')
    plt.ylabel('Probability')
    plt.xticks(range(10))
    
    plt.tight_layout()
    plt.savefig('src/custom_prediction.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("💾 Saved custom prediction to src/custom_prediction.png")

if __name__ == "__main__":
    print("🎯 Building and Training a Neural Network for Handwritten Digit Recognition")
    print("="*70)
    print(f"TensorFlow version: {tf.__version__}")
    
    # 1. Load and preprocess data
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()
    
    # 2. Build model
    print("\n🏗️  Building model...")
    model = build_model_sequential((28, 28), 10)
    model.summary()
    
    # 3. Train model
    history = train_model(model, (x_train, y_train), (x_test, y_test), epochs=5)
    
    # 4. Plot training history
    try:
        plot_training_history(history)
    except Exception as e:
        print(f"Training history plot skipped: {e}")
    
    # 5. Evaluate model
    test_loss, test_accuracy = evaluate_model(model, (x_test, y_test))
    
    # 6. Make some predictions
    indices, predicted_classes, predictions = make_predictions(model, (x_test, y_test))
    
    # 7. Visualize predictions (optional - requires matplotlib)
    try:
        visualize_predictions((x_test, y_test), indices, predicted_classes, predictions)
    except ImportError:
        print("📝 Install matplotlib to see prediction visualizations: pip install matplotlib")
    except Exception as e:
        print(f"Visualization skipped: {e}")
    
    # 8. Test with custom image
    try:
        test_custom_prediction(model)
    except Exception as e:
        print(f"Custom prediction test skipped: {e}")
    
    # 9. Save the trained model
    save_model(model)
    
    # 10. Demo loading the model
    print("\n🔄 Testing model loading...")
    loaded_model = load_saved_model()
    if loaded_model:
        print("✅ Model loading works! You can now use this model later.")
    
    print(f"\n🎉 Training complete! Final accuracy: {test_accuracy:.4f}")
    print("🚀 Next steps you could try:")
    print("   • Experiment with different architectures")
    print("   • Try different datasets (CIFAR-10, Fashion-MNIST)")
    print("   • Add more layers or change activation functions")
    print("   • Create your own handwritten digit images to test")
    print("   • Build a web interface for digit recognition")
    