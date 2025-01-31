# Image Classification using CNNs with Keras and TensorFlow
This project demonstrates how to build and train a Convolutional Neural Network (CNN) for image classification using the CIFAR-10 dataset. The model is implemented using Keras and TensorFlow.

## Steps

### Install Required Libraries:
```bash
pip install tensorflow numpy matplotlib
```

### Import Libraries:
```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import matplotlib.pyplot as plt
```

### Load and Preprocess Data:
The CIFAR-10 dataset is loaded and normalized.

Labels are converted to one-hot encoding.

```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)
```

### Build the CNN Model:
The model consists of:
- 3 Convolutional layers with ReLU activation.
- 2 MaxPooling layers.
- 2 Fully Connected (Dense) layers.

```python
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])
```

### Compile the Model:
- Optimizer: Adam.
- Loss Function: Categorical Crossentropy.
- Metrics: Accuracy.

```python
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

### Train the Model:
Trained for 10 epochs with a batch size of 64.

```python
history = model.fit(x_train, y_train, epochs=10, batch_size=64, validation_split=0.2)
```

### Evaluate the Model:
Test accuracy: 68.42%.

```python
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"Test accuracy: {test_acc}")
```

### Visualize Results:
Plots for training/validation accuracy and loss.

```python
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
```

### Make Predictions:
Predict and display sample images with their predicted labels.

```python
predictions = model.predict(x_test)
predicted_labels = np.argmax(predictions, axis=1)
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.figure(figsize=(10, 10))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.imshow(x_test[i])
    plt.title(f"Predicted: {class_names[predicted_labels[i]]}")
    plt.axis('off')
plt.show()
```

### Save the Model:
The trained model is saved as `cifar10_cnn_model.h5`.

```python
model.save("cifar10_cnn_model.h5")
```

## Results
- Training Accuracy: ~74.10%
- Validation Accuracy: ~68.88%
- Test Accuracy: ~68.42%

## Dependencies
- Python 3.7+
- TensorFlow 2.x
- NumPy
- Matplotlib

## How to Run

### Clone the repository:
```bash
git clone https://github.com/your-username/your-repo-name.git
```

### Install dependencies:
```bash
pip install tensorflow numpy matplotlib
```

### Run the notebook:
```bash
jupyter notebook your-notebook-file.ipynb
```

## License
This project is licensed under the MIT License. See the LICENSE file for details.
