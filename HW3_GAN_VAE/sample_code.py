import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os
import numpy as np

import matplotlib.pyplot as plt
figsize = 32
latent_dim = 3

### The code that can plot like figure 1
def plot_latent_space(vae, n=5):
    # display an n*n 2D manifold of digits
    digit_size = figsize
    scale = 1.0
    figure = np.zeros((digit_size * n, digit_size * n, 3))
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
            digit = x_decoded[0].reshape(digit_size, digit_size, 3)
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
    # plt.show()
    plt.savefig('out.png')
    
    return

### Load dataset from folder
dataset = keras.utils.image_dataset_from_directory(
    "./dataset_flower/sunflower/", label_mode=None, seed=123, image_size=(figsize, figsize), batch_size=32
)
dataset = dataset.map(lambda x: x/255.0)

### Show the figure
for x in dataset:
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    break
