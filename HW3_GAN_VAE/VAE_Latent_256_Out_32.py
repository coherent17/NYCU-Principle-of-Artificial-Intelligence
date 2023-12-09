import os
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# Set the path to your sunflower dataset
dataset_path = './dataset_flowers/sunflower'

# Output directory to save generated images and training plots
output_dir = './VAE_Latent_256_Out_32_Generated_Images'
os.makedirs(output_dir, exist_ok=True)

# Define image dimensions
figsize = 32
channels = 3

# Load images using image_dataset_from_directory
batch_size = 32
dataset = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    label_mode=None,
    seed=123,
    image_size=(figsize, figsize),
    batch_size=batch_size,
)
# Normalize pixel values to [0, 1]
dataset = dataset.map(lambda x: x / 255.0)

# Extract the first batch of images for training
sunflower_images = next(iter(dataset))

# Set the latent dimension (noise vector size)
latent_dim = 256

# VAE Model
class Sampling(layers.Layer):
    def call(self, inputs):
        mean, log_var = inputs
        batch = tf.shape(mean)[0]
        dim = tf.shape(mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return mean + tf.exp(0.5 * log_var) * epsilon

class VAE(models.Model):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.build_encoder()
        self.decoder = self.build_decoder()

    def build_encoder(self):
        encoder_inputs = layers.Input(shape=(figsize, figsize, channels))
        x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(32, activation="relu")(x)
        mean = layers.Dense(self.latent_dim, name="mean")(x)
        log_var = layers.Dense(self.latent_dim, name="log_var")(x)
        z = Sampling()([mean, log_var])
        encoder = models.Model(encoder_inputs, [mean, log_var, z])
        return encoder

    def build_decoder(self):
        latent_inputs = layers.Input(shape=(self.latent_dim,))
        x = layers.Dense(32 * 8 * 8, activation="relu")(latent_inputs)  # Adjusted for 32x32 output
        x = layers.Reshape((8, 8, 32))(x)
        x = layers.Conv2DTranspose(128, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Conv2DTranspose(64, 3, activation="relu", strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(0.2)(x)
        decoder_outputs = layers.Conv2DTranspose(channels, 3, activation="sigmoid", padding="same")(x)
        decoder = models.Model(latent_inputs, decoder_outputs)
        decoder.summary()
        return decoder

    def call(self, inputs):
        mean, log_var, z = self.encoder(inputs)
        reconstructed = self.decoder(z)
        return reconstructed

vae = VAE(latent_dim)
vae.build((None, figsize, figsize, channels))
vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001))

# Function to calculate VAE loss
def vae_loss(inputs, outputs, mean, log_var):
    reconstruction_loss = tf.reduce_mean(tf.square(inputs - outputs))
    kl_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mean) - tf.exp(log_var))
    return reconstruction_loss + kl_loss

### The code that can plot like figure 1
def plot_latent_space(vae, n=5):
    # display an n*n 2D manifold of digits
    digit_size = figsize
    scale = 5
    figure = np.zeros((digit_size * n, digit_size * n, channels))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-scale, scale, n)
    grid_y = np.linspace(-scale, scale, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            rest_latent = np.ones((1,latent_dim-2))*yi
            z_sample = np.array([[xi, yi]])
            z_sample = np.concatenate((z_sample, rest_latent), axis=1)
            x_decoded = vae.decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size, channels)
            figure[
                i * digit_size : (i + 1) * digit_size,
                j * digit_size : (j + 1) * digit_size,
            ] = digit

    plt.figure(figsize=(figsize, figsize))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure)
    plt.savefig(os.path.join(output_dir, 'different_latent.png'))
    plt.close()

# Training loop
epochs = 100
vae_loss_history = []
predict_images = []
test_input = tf.random.normal(shape=(1, latent_dim), seed=1)
for epoch in range(epochs + 1):
    for batch in dataset:
        # Forward pass
        with tf.GradientTape() as tape:
            mean, log_var, z = vae.encoder(batch)
            reconstructed_images = vae.decoder(z)
            loss = vae_loss(batch, reconstructed_images, mean, log_var)

        # Backward pass
        gradients = tape.gradient(loss, vae.trainable_weights)
        vae.optimizer.apply_gradients(zip(gradients, vae.trainable_weights))

    vae_loss_history.append(loss.numpy())
    print(f"Epoch {epoch + 1}, VAE Loss: {loss.numpy()}")

    # Generate images and plot latent space every 10 epochs
    if epoch % 10 == 0 and epoch != 0:
        predictions = vae.decoder(test_input)
        predict_images.append(predictions)
        plt.imshow(predictions.numpy()[0])
        plt.axis('off')
        plt.savefig(os.path.join(output_dir, f'generated_image_epoch_{epoch}.png'))
        plt.close()

# Plot latent space
plot_latent_space(vae, n=5)

# subplot the images generated per 10 epoch
plt.figure(figsize=(10, 4))
for i in range(len(predict_images)):
    plt.subplot(2, 5, i + 1)
    plt.imshow(predict_images[i].numpy()[0])
    plt.axis('off')

plt.savefig(os.path.join(output_dir, f'generated_image_per_10_epoch.png'))
plt.close()

# Plot the training history
plt.plot(vae_loss_history, label='VAE Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training History')
plt.legend()
plt.savefig(os.path.join(output_dir, 'vae_training_history.png'))
plt.show()
plt.close()
