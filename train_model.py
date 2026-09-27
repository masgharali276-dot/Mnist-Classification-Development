import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Load MNIST Dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# 2. Preprocess Data (Normalization)
X_train = X_train / 255.0
X_test = X_test / 255.0

# 3. Build Artificial Neural Network (ANN)
model = models.Sequential([
    layers.Flatten(input_shape=(28, 28)),  # Input Layer (28x28 -> 784)
    layers.Dense(128, activation='relu'),   # Hidden Layer 1
    layers.Dense(64, activation='relu'),    # Hidden Layer 2
    layers.Dense(10, activation='softmax')  # Output Layer (0-9 Digits)
])

# 4. Compile Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train Model
print("Training ANN Model...")
model.fit(X_train, y_train, epochs=5, validation_data=(X_test, y_test))

# 6. Save Model
model.save("mnist_model.h5")
print("Model saved successfully as 'mnist_model.h5'")