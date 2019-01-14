import json
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from urllib.request import urlretrieve

import adv_example
import forward_model 
import plot_results

def predict_and_plot(img, sess):
    img_probs = forward_model.predict(img, sess)
    plot_results.plot(img, img_probs)

# setup
tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

# get image of cat
img_path, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
img = PIL.Image.open(img_path)


# check that classifier indeed identifies it as a cat
#image_class_probs = forward_model.predict(img, sess)
#plot_results.plot(img, image_class_probs)
predict_and_plot(img, sess)

# variables to use later, calculated from other functions
# logits created when running inception
# probs created wien running inception on image
# image is what's being processed - tenserflow variable
logits, probs, image = forward_model.get_logits_probs_image_tf(sess)

# get our adversarial example and evaluate it
adv_img = adv_example.generate_adversarial_example(img,sess)
#adv_class_probs = forward_model.predict(adv_img,sess)
#plot_results.plot(adv_img, adv_class_probs)
predict_and_plot(adv_img, sess)

# no need to make a robust example
def add_noise(image, noise_factor = 1):
    # adding gaussian noise
    mean = 0.0
    var = 2.0
    stdev = var**0.5
    w, h = image.size
    c = len(image.getbands())

    noise = np.random.normal(mean, stdev, (h, w, c))
    noisy_image = PIL.Image.fromarray(np.uint8(np.array(image) + noise))
    
    
    return noisy_image
    
# now add noise to the original image and see if it classifies correctly
noised_image = add_noise(img)
#noised_image_probs = forward_model.predict(noised_image, sess)
#plot_results.plot(noised_image, noised_image_probs)
predict_and_plot(noised_image, sess)

# now check adversarial, see if it's fooled
noised_adv_img = add_noise(adv_img)
#noised_adv_img_probs = forward_model.predict(noised_image, sess)
predict_and_plot(noised_adv_img, sess)
