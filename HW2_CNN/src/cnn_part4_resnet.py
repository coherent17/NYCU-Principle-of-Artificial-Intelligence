import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Activation, Add, GlobalAveragePooling2D, Dense, Dropout
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt

# Load the CIFAR-10 data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize the data
train_images, test_images = train_images / 255.0, test_images / 255.0

# Convert labels to one-hot encoding
num_classes = 10
train_labels_one_hot = to_categorical(train_labels, num_classes)
test_labels_one_hot = to_categorical(test_labels, num_classes)

# Define ResNet block
def resnet_block(x, filters, kernel_size=3, stride=1, use_dropout=False):
    shortcut = x  # Preserve the input for the shortcut connection

    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(x)
    y = BatchNormalization()(y)
    y = Activation('relu')(y)

    if use_dropout:
        y = Dropout(0.25)(y)

    y = Conv2D(filters, kernel_size=kernel_size, strides=stride, padding='same')(y)
    y = BatchNormalization()(y)

    # Check if the number of channels is the same, else apply 1x1 convolution
    if shortcut.shape[-1] != y.shape[-1]:
        shortcut = Conv2D(filters, kernel_size=1, strides=stride, padding='same')(shortcut)

    return Add()([shortcut, y])

# Build ResNet model
input_layer = Input(shape=(32, 32, 3))
conv1 = Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)

# Using two residual blocks
res_block1 = resnet_block(conv1, 32)
res_block2 = resnet_block(res_block1, 64, use_dropout=True)

# Global Average Pooling
gap = GlobalAveragePooling2D()(res_block2)

# Fully connected layers
fc1 = Dense(128, activation='relu')(gap)
fc1 = Dropout(0.5)(fc1)
output_layer = Dense(num_classes, activation='softmax')(fc1)

# Create the ResNet model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
model.summary()

# Train the model
batch_size = 128
epochs = 50

history = model.fit(train_images, train_labels_one_hot,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(test_images, test_labels_one_hot),
                    shuffle=True)

# Plot the loss value for each epoch
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plot the accuracy for each epoch
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate the model
score = model.evaluate(test_images, test_labels_one_hot, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predict the labels for the test set
y_pred_one_hot = model.predict(test_images)
y_pred_labels = np.argmax(y_pred_one_hot, axis=1)

# Convert one-hot encoded true labels to class labels
y_true_labels = np.argmax(test_labels_one_hot, axis=1)

# Generate the confusion matrix
conf_mat = confusion_matrix(y_true_labels, y_pred_labels)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
plt.imshow(conf_mat, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Confusion Matrix')
plt.colorbar()

classes = range(num_classes)
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)

plt.xlabel('Predicted Label')
plt.ylabel('True Label')

for i in range(len(classes)):
    for j in range(len(classes)):
        plt.text(j, i, format(conf_mat[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if conf_mat[i, j] > conf_mat.max() / 2. else "black")

plt.tight_layout()
plt.show()

# Print the classification report for more detailed evaluation
print("Classification Report:\n", classification_report(y_true_labels, y_pred_labels))