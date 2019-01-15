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
#adv_img = adv_example.generate_adversarial_example(img,sess)
adv_img = PIL.Image.open('adversarial_cat.png')


# save image for later use (don't generate every time, for testing)
#adv_img.save('adversarial_cat.png')

#adv_class_probs = forward_model.predict(adv_img,sess)
#plot_results.plot(adv_img, adv_class_probs)
predict_and_plot(adv_img, sess)

#robust_adv_img = adv_example.generate_adversarial_example(img, sess, mode="rot_robust")
robust_adv_img = PIL.Image.open('robust_adversarial_cat.png')

# save robust image for later use
#robust_adv_img.save('robust_adversarial_cat.png')

# no need to make a robust example
def add_noise(image, noise_factor=1, noise_type="gauss"):
    noisy_image = None
    image_array = np.array(image)
    if noise_type == "gauss":
        # adding gaussian noise
        mean = 0.0
        var = 1.0 * noise_factor
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
        #w, h = image.size
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
        # set noisy image to original image? or return none?
    
    return noisy_image

def test_with_noise(images, noise_factor, noise_type, sess):
    for image in images:
        noised_image = add_noise(image, noise_factor=noise_factor,
                                 noise_type=noise_type)
        predict_and_plot(noised_image, sess)

images = [img, adv_img, robust_adv_img]
test_with_noise(images, 1, "speckle", sess)

"""   
# now add noise to the original image and see if it classifies correctly
noised_image = add_noise(img)
predict_and_plot(noised_image, sess)

# now check adversarial, see if it's fooled
noised_adv_img = add_noise(adv_img)
predict_and_plot(noised_adv_img, sess)

# now check robust
noised_robust = add_noise(robust_adv_img, noise_factor=1)
predict_and_plot(noised_robust, sess)
"""

# test effect of multiple different noise factors
#for i in np.linspace(0,1,11):
#    print(f"Testing noise_factor of {i}")
#    test_with_noise(images, i, "speckle", sess)
