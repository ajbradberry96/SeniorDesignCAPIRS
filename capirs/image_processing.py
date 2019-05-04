# -*- coding: utf-8 -*-
"""
Module for image transforms.

rand_warp: perspective warp
add_noise: various options to add noise
saturate_mod: image saturation
color_shift: changes image colors
gaussian_blur: applies gaussian blur
random_blur: applies random blur
box_blur: applies box blur

@author: Andrew Bradberry, Brian Greber, Drew Harbor, Joshua Osborne, Chris Rodriguez
"""
import logging
import PIL
import numpy as np
import random


def find_coeffs(pa, pb):
    """ Finds coefficients for perspective shift for use in rand_warp() """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0,
                       0, 0, -p2[0] * p1[0], -p2[0] * p1[1]])
        matrix.append([0, 0, 0, p1[0],
                       p1[1], 1, -p2[1] * p1[0], -p2[1] * p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)


def rand_warp(image, deg=None, random_state=None):
    """
    Performs a random perspective warp of the original image

    :param image: PIL image to warp
    :param deg: degree to warp image
    default is None, which sets degree to a proportion of the image size
    :param random_state: seed to map points on original image
    default is None
    :return: the modified PIL image
    """

    if deg is None:
        deg = image.size[0] // 20

    ht = image.height
    wt = image.width

    np.random.seed(random_state)
    # Points on the original image to be mapped to...
    pa = [(np.random.randn() * deg, np.random.randn() * deg),
          (wt - np.random.randn() * deg, np.random.randn() * deg),
          (wt - np.random.randn() * deg, ht - np.random.randn() * deg),
          (np.random.randn() * deg, ht - np.random.randn() * deg)]
    # Points on the output image
    pb = [(np.random.randn() * deg, np.random.randn() * deg),
          (wt - np.random.randn() * deg, np.random.randn() * deg),
          (wt - np.random.randn() * deg, ht - np.random.randn() * deg),
          (np.random.randn() * deg, ht - np.random.randn() * deg)]

    return image.transform(
        image.size, PIL.Image.PERSPECTIVE,
        find_coeffs(pa, pb),
        PIL.Image.BILINEAR)


def add_noise(image, noise_factor=0.1, noise_type="speckle"):
    """
    Add noise to an image to defeat adversarial examples.

    :param image: PIL image to add noise to
    :param noise_factor: intensity to scale noise, different for each noise type
    :param noise_type:which method to add noise. Takes in a string, four choices:
        1. gauss - gaussian noise, somewhat effective at noise_factor of 1
        2. saltpepper - salt and pepper noise, somewhat effective at
            noise_factor of 1
        3. poisson - not effective, noise_factor does not affect it
        4. speckle - very effective, best at noise_factor of ~0.3
    :return: the modified PIL image with noise added to it.
    """

    noisy_image = None
    image_array = np.array(image)
    if noise_type == "gauss":
        # adding gaussian noise
        mean = 0.0
        var = 5.0 * noise_factor  # intensity of gauss differences
        stdev = var ** 0.5
        w, h = image.size
        c = len(image.getbands())

        noise = np.random.normal(mean, stdev, (h, w, c))
        noisy_image = PIL.Image.fromarray(np.uint8(np.array(image) + noise))
    elif noise_type == "saltpepper":
        noisy_image = image_array
        # add salt and pepper noise
        s_vs_p = 0.5  # ratio of salt to pepper
        amount = 0.004 * noise_factor
        # generate salt (1) noise
        numpixels = image_array.size
        num_salt = np.ceil(amount * numpixels * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image_array.shape]
        noisy_image[tuple(coords)] = 255
        # generate pepper (0) noise
        num_pepper = np.ceil(amount * numpixels * (1 - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image_array.shape]
        noisy_image[tuple(coords)] = 0
        noisy_image = PIL.Image.fromarray(np.uint8(np.array(noisy_image)))
    elif noise_type == "poisson":
        vals = len(np.unique(image_array))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image_array * vals) / float(vals)
        noisy_image = PIL.Image.fromarray(np.uint8(np.array(noisy)))
    elif noise_type == "speckle":
        row, col, ch = image_array.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image_array + image_array * gauss * noise_factor
        noisy_image = PIL.Image.fromarray(np.uint8(np.array(noisy)))
    else:
        logging.warning("Invalid noise type specified")
        # set noisy image to original image? or return none? raise exception?

    return noisy_image


def saturate_mod(img, saturate_amt=0):
    """
    Takes a PIL image and saturates it

    :param img: PIL image to saturate
    :param saturate_amt: how far to saturate the image;
    if set to 0, saturates randomly between 0.5 and 2
    :return: the modified PIL image
    """

    if saturate_amt == 0:
        saturate_amt = random.uniform(9, 10)

    converter = PIL.ImageEnhance.Color(img)
    img2 = converter.enhance(saturate_amt)
    return img2


def color_shift(img, shifts=(0, 0, 0)):
    """
    Takes a PIL image and shifts the color some amount

    :param img: PIL image to color shift
    :param shifts: 3-item tuple of (R, G, B) shift amount;
    default is (0,0,0), which indicates random shift amounts
    :return: the modified PIL image
    """

    if shifts[0] == 0 and shifts[1] == 0 and shifts[2] == 0:
        red_shift = random.randint(190, 200)
        green_shift = random.randint(190, 200)
        blue_shift = random.randint(190, 200)
        shifts = (red_shift, green_shift, blue_shift)

    arr = np.asarray(img).copy()
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            # Loop through all pixels and shift them by the given amounts
            new_red = arr[i, j, 0] + shifts[0]
            if new_red >= 255:
                new_red = 255
            arr[i, j, 0] = new_red

            new_green = arr[i, j, 1] + shifts[1]
            if new_green >= 255:
                new_green = 255
            arr[i, j, 1] = new_green

            new_blue = arr[i, j, 2] + shifts[2]
            if new_blue >= 255:
                new_blue = 255
            arr[i, j, 2] = new_blue

    return PIL.Image.fromarray(np.uint8(arr))


def gaussian_blur(image, radius):
    """Gaussian Blur function, takes in image and radius of kernel"""
    blur = image.filter(PIL.ImageFilter.GaussianBlur(radius))
    return blur


def random_blur(image, size):
    """
    Create a random kernel convolution
    size must be 3 or 5
    """
    k = []
    for i in range(size ** 2):
        k.append(random.randint(1, 30))
    blur = image.filter(PIL.ImageFilter.Kernel((size, size), k))
    return blur


def box_blur(image, radius):
    """Box blur, takes in image and radius of kernel"""
    blur = image.filter(PIL.ImageFilter.BoxBlur(radius))
    return blur
