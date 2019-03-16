import tensorflow as tf

import forward_model
import plot_results
import adv_example
import image_processing
import detect_adversarial
import image_processing
import sys

from urllib.request import urlretrieve

import PIL

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

img = PIL.Image.open("media/norm_image_raw.jpg")
img = img.resize((300, 300), PIL.Image.ANTIALIAS)
img.save("media/norm_img.png")

forward_model.init(sess)

print(forward_model.get_imagenet_labels())

adv_img = adv_example.generate_adversarial_example(img, sess, adv_class='jellyfish')

adv_img.save("media/adv_img.png")



input("Continue? ")

img = PIL.Image.open("media/norm_img.png")
adv_img = PIL.Image.open("media/adv_img.png")

#img = PIL.Image.open("media/cat.png")
#adv_img = PIL.Image.open("media/adversarial_cat.png")


norm_probs = forward_model.predict(img)
adv_probs = forward_model.predict(adv_img)

plot_results.plot(img,norm_probs)
plot_results.plot(adv_img,adv_probs)
plot_results.plt.show()


input("Continue? ")


img = PIL.Image.open("media/norm_img.png")
adv_img = PIL.Image.open("media/adv_img.png")

#img = PIL.Image.open("media/cat.png")
#adv_img = PIL.Image.open("media/adversarial_cat.png")


print("NORMAL IMAGE: ")
detect_adversarial.detect(img)
print()
print()

plot_results.plt.show()

print("ADVERSARIAL IMAGE: ")
detect_adversarial.detect(adv_img)
print()
print()

plot_results.plt.show()


