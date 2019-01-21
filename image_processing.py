# -*- coding: utf-8 -*-
"""
Module for image transforms.

@author: Andrew Bradberry
"""
import PIL
from urllib.request import urlretrieve
import numpy as np

def find_coeffs(pa, pb):
    """ Finds coefficients for perspective shift for use in rand_warp() """
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def rand_warp(image, deg=None, random_state=None):
    """Performs a random perspective warp of the original image"""
    if deg == None:
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


def add_noise(image, noise_factor=0.3, noise_type="speckle"):
    """Add noise to an image to defeat adversarial examples.

    Positional arguments:
    image - PIL image to add noise to

    Keyword arguments:
    noise_factor: intensity to scale noise, different for each noise type
    noise_type: which method to add noise. Takes in a string, four choices:
        1. gauss - gaussian noise, somewhat effective at noise_factor of 1
        2. saltpepper - salt and pepper noise, somewhat effective at
            noise_factor of 1
        3. poisson - not effective, noise_factor does not affect it
        4. speckle - very effective, best at noise_factor of ~0.3

    Returns the PIL image with noise added to it.
    """
    noisy_image = None
    image_array = np.array(image)
    if noise_type == "gauss":
        # adding gaussian noise
        mean = 0.0
        var = 1.0 * noise_factor # intensity of gauss differences
        stdev = var**0.5
        w, h = image.size
        c = len(image.getbands())

        noise = np.random.normal(mean, stdev, (h, w, c))
        noisy_image = PIL.Image.fromarray(np.uint8(np.array(image) + noise))
    elif noise_type == "saltpepper":
        noisy_image = image_array
        # add salt and pepper noise
        s_vs_p = 0.5 # ratio of salt to pepper
        amount = 0.004 * noise_factor
        # generate salt (1) noise
        numpixels = image_array.size
        num_salt = np.ceil(amount * numpixels * s_vs_p)
        coords = [np.random.randint(0, i-1, int(num_salt)) for i in image_array.shape]
        noisy_image[tuple(coords)] = 255
        # generate pepper (0) noise
        num_pepper = np.ceil(amount * numpixels * (1 - s_vs_p))
        coords = [np.random.randint(0, i-1, int(num_pepper)) for i in image_array.shape]
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
        # TODO: replace with log statement 
        print("Invalid noise type specified")
        # set noisy image to original image? or return none? raise exception?
    
    return noisy_image
