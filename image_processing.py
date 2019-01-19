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

