import tensorflow as tf

import forward_model
import plot_results
import adv_example
<<<<<<< HEAD
<<<<<<< Updated upstream
=======
import image_processing
import detect_adversarial
>>>>>>> Stashed changes
=======
import image_processing
>>>>>>> 781903840b87b92411d507c210f317a8ca0c263f

from urllib.request import urlretrieve

import PIL

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

img_path, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/cat.jpg')
img = PIL.Image.open(img_path)

forward_model.init(sess)
detect_adversarial.detect(img)

adv_img = PIL.Image.open("media/robust_adversarial_cat.png")
detect_adversarial.detect(adv_img)

# TODO: refactor this whole mess
sys.exit()


image_class_probs = forward_model.predict(img, sess)
plot_results.plot(img, image_class_probs)

logits, probs, image = forward_model.get_logits_probs_image_tf(sess)

#adv_img = adv_example.generate_adversarial_example(img,sess)
adv_img = PIL.Image.open("media/robust_adversarial_cat.png")
adv_class_probs = forward_model.predict(adv_img,sess)
plot_results.plot(adv_img, adv_class_probs)

col_img = image_processing.saturate_mod(image_processing.color_shift(adv_img))
col_class_probs = forward_model.predict(col_img, sess)
plot_results.plot(col_img, col_class_probs)

adv_robust = adv_example.generate_adversarial_example(img, sess, mode="rot_robust")
#ex_angle = np.pi/8
#angle = tf.placeholder(tf.float32, ())
#rotated_image = tf.contrib.image.rotate(image, angle)
#rotated_example = rotated_image.eval(feed_dict={image: (np.asarray(adv_robust)/255), angle: ex_angle})
#rot_img = PIL.Image.fromarray(np.uint8((rotated_example)*255))
rot_img = adv_robust.rotate(45, expand=0)
rot_class_probs = forward_model.predict(rot_img, sess)
plot_results.plot(rot_img, rot_class_probs)
