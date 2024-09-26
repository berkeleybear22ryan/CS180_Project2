import cv2
import numpy as np
import scipy as sp
import tensorflow as tf
import os


# will just make sure to normalize everything for this project
# a = normalize(image)
def normalize(image):
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


# image = open_image(image_path)
def open_image(path):
    # will always open the image as default even if black and white to have consistency
    image = cv2.imread(path)
    return image


# save_image(output_path+name, a)
def save_image(path, image):
    cv2.imwrite(path, image)


# same_image(image, a)
def same_image(img1, img2):
    return np.all(img1 == img2)


def convolve_v1(img, filter, mode="same", boundary='fill'):
    print(f"img.shape: {img.shape}")
    print(f"filter.shape: {filter.shape}")
    result = sp.signal.convolve2d(img, filter, mode=mode, boundary=boundary, fillvalue=0)
    print(f"result.shape: {result.shape}")
    print("\n")
    return result


# supports multiple channels now
def convolve_v3(img, filter, mode="same", boundary='fill'):
    print(f"img.shape: {img.shape}")
    print(f"filter.shape: {filter.shape}")
    # now check if the image has multiple channels
    if len(img.shape) == 3:
        # multi-channel image (e.g., RGB)
        channels = []
        for i in range(img.shape[2]):
            # apply convolution to each channel separately
            convolved_channel = sp.signal.convolve2d(img[:, :, i], filter, mode=mode, boundary=boundary, fillvalue=0)
            channels.append(convolved_channel)
        # stack the channels back together
        result = np.stack(channels, axis=-1)
    else:
        # single-channel image
        result = sp.signal.convolve2d(img, filter, mode=mode, boundary=boundary, fillvalue=0)
    print(f"result.shape: {result.shape}")
    print("\n")
    return result


# this is the tensorflow version which shoud be faster + has color footprint
def convolve_v2(img, filter):
    print(f"img.shape: {img.shape}")
    print(f"filter.shape: {filter.shape}")
    # make sure to convert input to TensorFlow tensor if it's not already
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    filter = tf.convert_to_tensor(filter, dtype=tf.float32)
    # now flip the filter horizontally and vertically you can use this to modify result or combine both
    # derivate from top + bottom and left + right
    filter = tf.reverse(filter, axis=[1])
    filter = tf.reverse(filter, axis=[0])
    # and reshape filter to [height, width, in_channels, out_channels]
    filter = tf.reshape(filter, (*filter.shape, 1, 1))
    # split the image into separate channels
    channels = tf.unstack(img, axis=-1)

    # apply convolution to each channel separately
    convolved_channels = []
    for channel in channels:
        # add batch and channel dimensions
        channel = tf.expand_dims(tf.expand_dims(channel, axis=0), axis=-1)
        # perform convolution
        convolved = tf.nn.conv2d(channel, filter, strides=1, padding='SAME')
        # remove extra dimensions
        convolved = tf.squeeze(convolved, axis=[0, -1])
        convolved_channels.append(convolved)

    # stack the convolved channels back together
    result = tf.stack(convolved_channels, axis=-1)
    # convert result back to numpy array
    result_np = result.numpy()
    print(f"result.shape: {result_np.shape}")
    print("\n")
    return result_np


# combine using gradient magnitude i.e. for just images convolved on D_x, D_y or there reverses
def combine_images_gradient_magnitude(img1, img2):
    result = np.sqrt(img1 ** 2 + img2 ** 2)
    return result


def threshold_image(image, threshold):
    binary_image = (image > threshold).astype(np.uint8) * 255
    return binary_image

# FILTERS---------------------------------------------------------------------
D_x = np.array([[1, -1]])
D_y = np.array([[1], [-1]])

# sobel filter in the x-direction (S_x)
S_x = np.array([
    [1, 0, -1],
    [2, 0, -2],
    [1, 0, -1]
])

# sobel filter in the y-direction (S_y)
S_y = np.array([
    [1, 2, 1],
    [0, 0, 0],
    [-1, -2, -1]
])


# ksize = 5  # kernel size
# sigma = 1.0  # standard deviation
def gaussian_filter(ksize=5, sigma=1.0):
    # create 1D Gaussian kernel
    kernel_1d = cv2.getGaussianKernel(ksize, sigma)
    # create 2D Gaussian kernel by taking the outer product
    gaussian_kernel = np.outer(kernel_1d, kernel_1d)
    print("Gaussian Kernel (G):\n", gaussian_kernel)
    return gaussian_kernel


G = gaussian_filter(10, 2)

# FILTERS---------------------------------------------------------------------
