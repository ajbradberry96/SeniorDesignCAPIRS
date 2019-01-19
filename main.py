import tensorflow as tf

import forward_model
import plot_results
import adv_example

from urllib.request import urlretrieve

import PIL

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

img_path, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
img = PIL.Image.open(img_path)

image_class_probs = forward_model.predict(img, sess)
plot_results.plot(img, image_class_probs)

logits, probs, image = forward_model.get_logits_probs_image_tf(sess)

adv_img = adv_example.generate_adversarial_example(img,sess)
adv_class_probs = forward_model.predict(adv_img,sess)
plot_results.plot(adv_img, adv_class_probs)

adv_robust = adv_example.generate_adversarial_example(img, sess, mode="rot_robust")
#ex_angle = np.pi/8
#angle = tf.placeholder(tf.float32, ())
#rotated_image = tf.contrib.image.rotate(image, angle)
#rotated_example = rotated_image.eval(feed_dict={image: (np.asarray(adv_robust)/255), angle: ex_angle})
#rot_img = PIL.Image.fromarray(np.uint8((rotated_example)*255))
rot_img = adv_robust.rotate(45, expand=0)
rot_class_probs = forward_model.predict(rot_img, sess)
plot_results.plot(rot_img, rot_class_probs)
