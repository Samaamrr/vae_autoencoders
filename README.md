# VAE on Simple Autoencoder

This repository contains implementations of Variational Autoencoders (VAEs) for different datasets and architectures. VAEs are a type of neural network architecture used for unsupervised learning of latent representations of data.

## Table of Contents
1. [Introduction](#introduction)
2. [Datasets](#datasets)
3. [Architectures](#architectures)
4. [Training](#training)
5. [Results](#results)
6. [Usage](#usage)
7. [References](#references)

## Introduction

Variational Autoencoders (VAEs) consist of an encoder network, a decoder network, and a loss function that encourages the learned latent space to follow a specific distribution, typically a Gaussian distribution. This repository demonstrates how to implement and train VAEs on the MNIST and Fashion MNIST datasets, as well as deeper autoencoder architectures.

## Datasets

- **MNIST**: Handwritten digit dataset containing 60,000 training and 10,000 test images.
- **Fashion MNIST**: Zalando's article images consisting of 60,000 training and 10,000 test images.

## Architectures

### 1. Simple Autoencoder
A simple VAE architecture with a 2-dimensional latent space.

### 2. Deep Autoencoder
A deeper VAE architecture with a latent space of varying dimensions (2 and 4).

## Training

The models are trained using the Adam optimizer and a custom VAE loss function, which combines reconstruction loss and KL divergence to regularize the latent space.

### VAE Loss Function
```python
def vae_loss(inputs, outputs):
    reconstruction_loss = losses.binary_crossentropy(tf.keras.backend.flatten(inputs), tf.keras.backend.flatten(outputs)) * 28 * 28
    z_mean, z_log_var, _ = encoder(inputs)
    kl_loss = -0.5 * tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=-1)
    return tf.reduce_mean(reconstruction_loss + kl_loss)

