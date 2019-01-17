# -*- coding: utf-8 -*-
"""
Module for image transforms.

@author: Andrew Bradberry
"""
import PIL
from urllib.request import urlretrieve
import numpy as np
"""
import tensorflow as tf
import forward_model
import plot_results
import adv_example
"""
def find_coeffs(pa, pb):
    matrix = []
    for p1, p2 in zip(pa, pb):
        matrix.append([p1[0], p1[1], 1, 0, 0, 0, -p2[0]*p1[0], -p2[0]*p1[1]])
        matrix.append([0, 0, 0, p1[0], p1[1], 1, -p2[1]*p1[0], -p2[1]*p1[1]])

    A = np.matrix(matrix, dtype=np.float)
    B = np.array(pb).reshape(8)

    res = np.dot(np.linalg.inv(A.T * A) * A.T, B)
    return np.array(res).reshape(8)

def rand_warp(image, deg=None, random_state=None):
    if deg == None:
        deg = image.size[0] // 20
        
    ht = image.height
    wt = image.width
    
    np.random.seed(random_state)
    pa = [(np.random.randn() * deg, np.random.randn() * deg), 
          (wt - np.random.randn() * deg, np.random.randn() * deg), 
          (wt - np.random.randn() * deg, ht - np.random.randn() * deg),
          (np.random.randn() * deg, ht - np.random.randn() * deg)]
    pb = [(np.random.randn() * deg, np.random.randn() * deg), 
          (wt - np.random.randn() * deg, np.random.randn() * deg), 
          (wt - np.random.randn() * deg, ht - np.random.randn() * deg),
          (np.random.randn() * deg, ht - np.random.randn() * deg)]
    
    return image.transform(
        image.size, PIL.Image.PERSPECTIVE,
        find_coeffs(pa, pb),
        PIL.Image.BILINEAR)
            
""" This was used for testing rand_warp... it works! 
img_path, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
img = PIL.Image.open(img_path)

tf.logging.set_verbosity(tf.logging.ERROR)

with tf.Session() as sess:
    image_class_probs = forward_model.predict(img, sess)
    plot_results.plot(img, image_class_probs)
    
    logits, probs, image = forward_model.get_logits_probs_image_tf(sess)
    
    adv_img = adv_example.generate_adversarial_example(img,sess)
    adv_class_probs = forward_model.predict(adv_img,sess)
    plot_results.plot(adv_img, adv_class_probs)
    
    for i in range(1, 11):
        warped_normal = rand_warp(img, random_state=i)
        image_class_probs = forward_model.predict(warped_normal, sess)
        plot_results.plot(warped_normal, image_class_probs)
        
        warped_adv = rand_warp(adv_img, random_state=i)
        image_class_probs = forward_model.predict(warped_adv, sess)
        plot_results.plot(warped_adv, image_class_probs)
