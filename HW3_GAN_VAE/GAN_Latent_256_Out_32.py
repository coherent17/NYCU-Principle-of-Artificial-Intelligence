import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Set the path to your flower dataset
dataset_path = './dataset_flowers'

# Output directory to save generated images and training plots
output_dir = './GAN_Latent_256_Out_32_Generated_Images'
os.makedirs(output_dir, exist_ok=True)

# Define image dimensions
img_height = 32
img_width = 32
channels = 3

# Create a flow from directory generator
flower_generator = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
flower_data = flower_generator.flow_from_directory(
    dataset_path,
    target_size=(img_height, img_width),
    batch_size=64,
    class_mode=None  # Since it's an unlabelled dataset for GAN
)

# Set the latent dimension (noise vector size)
latent_dim = 256

# Generator Model
def build_generator(latent_dim):
    model = models.Sequential()
    model.add(layers.Dense(128 * 8 * 8, input_dim=latent_dim))
    model.add(layers.Reshape((8, 8, 128)))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2DTranspose(128, (4, 4), strides=(2, 2), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(3, (3, 3), activation='tanh', padding='same'))  # Output shape: (32, 32, 3)
    return model

# Discriminator Model
def build_discriminator(img_shape):
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same', input_shape=img_shape))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU(alpha=0.2))
    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

# GAN Model
def build_gan(generator, discriminator):
    discriminator.trainable = False
    model = models.Sequential()
    model.add(generator)
    model.add(discriminator)
    return model

# Build and compile the discriminator
discriminator = build_discriminator((img_height, img_width, channels))
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5), metrics=['accuracy'])

# Build the generator
generator = build_generator(latent_dim)

# Build and compile the GAN model
discriminator.trainable = False
gan = build_gan(generator, discriminator)
gan.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5))

# Function to generate and save images
def generate_and_save_images(model, epoch, test_input):
    predictions = model.predict(test_input)
    num_generated_images = min(predictions.shape[0], 4)  # Limit to 16 images for simplicity

    plt.figure(figsize=(4, 4))

    for i in range(num_generated_images):
        plt.subplot(2, 2, i + 1)
        plt.imshow(predictions[i] * 0.5 + 0.5)  # Rescale images to [0, 1]
        plt.axis('off')

    plt.savefig(os.path.join(output_dir, f'generated_image_epoch_{epoch}.png'))
    plt.close()

# Training loop
epochs = 100
batch_size = 64
gen_loss_history = []
disc_loss_history = []
noise_input = tf.random.normal(shape=(1, latent_dim), seed = 1)
test_input = tf.random.normal(shape=(4, latent_dim), seed=1)
predict_images = []

for epoch in range(epochs):
    for _ in range(len(flower_data) - 1):
        # Train Discriminator
        noise = tf.random.normal(shape=(batch_size, latent_dim))
        generated_images = generator.predict(noise)
        real_images = next(flower_data)
        
        # Ensure the number of samples in real_images matches batch_size
        if(real_images.shape[0] != batch_size):
            continue
        real_images = real_images[:batch_size]

        # Combine real and fake images
        all_images = np.concatenate([real_images, generated_images])

        # Combine real and fake labels
        labels_real = tf.ones((batch_size, 1)) * 0.9    # Prevent discriminator be too confident
        labels_fake = tf.zeros((batch_size, 1))
        all_labels = np.concatenate([labels_real, labels_fake])

        # Shuffle the combined data
        indices = np.arange(2 * batch_size)
        np.random.shuffle(indices)
        shuffled_images = all_images[indices]
        shuffled_labels = all_labels[indices]

        d_loss = discriminator.train_on_batch(shuffled_images, shuffled_labels)

        # Train Generator
        noise = tf.random.normal(shape=(batch_size, latent_dim))
        labels_gan = tf.ones((batch_size, 1))

        g_loss = gan.train_on_batch(noise, labels_gan)

    gen_loss_history.append(g_loss)
    disc_loss_history.append(d_loss[0])

    # generate images every 10 epochs
    if epoch % 10 == 0:
        predict_images.append(generator.predict(noise_input))

    print(f"Epoch {epoch + 1}, D Loss: {d_loss[0]}, G Loss: {g_loss}")
    generate_and_save_images(generator, epoch + 1, test_input)




# subplot the images generated per 10 epoch
plt.figure(figsize=(10, 4))
for i in range(len(predict_images)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(np.reshape(predict_images[i] * 0.5 + 0.5, (32, 32, 3)))  # Rescale images to [0, 1]
    plt.axis('off')

plt.savefig(os.path.join(output_dir, f'generated_image_per_10_epoch.png'))
plt.close()


# Plot the training history
plt.plot(gen_loss_history, label='Generator Loss')
plt.plot(disc_loss_history, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.savefig(os.path.join(output_dir, 'training_history.png'))
plt.close()