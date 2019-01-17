import json
import matplotlib.pyplot as plt
import numpy as np

from PIL import ImageFilter

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

### get image of cat
##img_path, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
##img = PIL.Image.open(img_path)
##
##img.save('Cat.png')
img = PIL.Image.open('Cat.png')

#classify image
predict_and_plot(img, sess)

# variables to use later, calculated from other functions
# logits created when running inception
# probs created wien running inception on image
# image is what's being processed - tenserflow variable
logits, probs, image = forward_model.get_logits_probs_image_tf(sess)

### get adv example
##advImg = adv_example.generate_adversarial_example(img, sess)
##advImg.save('adversarialCat.png')
advImg = PIL.Image.open('adversarialCat.png')

# classify adversarial example
predict_and_plot(advImg, sess)

### get robust adv example
##robustadvImg = adv_example.generate_adversarial_example(img, sess, mode='rot_robust')
##robustadvImg.save('rotRobustAdversarialCat.png')
robustadvImg = PIL.Image.open('rotRobustAdversarialCat.png')

#classify the robust example
predict_and_plot(robustadvImg, sess)


## Gaussian Blur function, takes in image and radius of kernal
def gaussianBlur(image, radius):
    blur = image.filter(ImageFilter.GaussianBlur(radius))
    return blur

# Blur the 3 test images with radius 2
imgBlur =  gaussianBlur(img, 2)
advImgBlur = gaussianBlur(advImg, 2)
robustadvImgBlur = gaussianBlur(robustadvImg, 2)
# Classify the Blurred images
predict_and_plot(imgBlur, sess)
predict_and_plot(advImgBlur, sess) #defeated the example
predict_and_plot(robustadvImgBlur, sess) #defeated the example



