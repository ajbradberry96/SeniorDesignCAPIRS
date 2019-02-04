import tensorflow as tf
import tensorflow.contrib.slim as slim
import tensorflow.contrib.slim.nets as nets

import tempfile
from urllib.request import urlretrieve
import tarfile
import os
import json

import numpy as np


def inception(image, reuse):
	preprocessed = tf.multiply(tf.subtract(tf.expand_dims(image, 0), 0.5), 2.0)
	arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
	with slim.arg_scope(arg_scope):
		logits, _ = nets.inception.inception_v3(
			preprocessed, 1001, is_training=False, reuse=reuse)
		logits = logits[:,1:] # ignore background class
		probs = tf.nn.softmax(logits) # probabilities
	return logits, probs


def load_checkpoint(sess):
	data_dir = tempfile.mkdtemp()
	inception_tarball, _ = urlretrieve('http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz')
	tarfile.open(inception_tarball, 'r:gz').extractall(data_dir)

	restore_vars = [
		var for var in tf.global_variables()
		if var.name.startswith('InceptionV3/')
	]
	saver = tf.train.Saver(restore_vars)
	saver.restore(sess, os.path.join(data_dir, 'inception_v3.ckpt'))

def get_imagenet_labels():
	imagenet_json, _ = urlretrieve('http://www.anishathalye.com/media/2017/07/25/imagenet.json')
	with open(imagenet_json) as f:
		imagenet_labels = json.load(f)
	return imagenet_labels

def image_preprocessor(img):
	wide = img.width > img.height
	new_w = 299 if not wide else int(img.width * 299 / img.height)
	new_h = 299 if wide else int(img.height * 299 / img.width)
	img = img.resize((new_w, new_h)).crop((0, 0, 299, 299))
	img = (np.asarray(img) / 255.0).astype(np.float32)
	return img

model_loaded = False
logits = None
probs = None
image = None
sess = None

def init(tf_sess):
	global logits, probs, image, model_loaded, sess
	sess = tf_sess
	image = tf.Variable(tf.zeros((299, 299, 3)))
	logits, probs = inception(image, reuse=False)
	load_checkpoint(sess)
	model_loaded = True


def predict(input_image):
	global logits, probs, image, model_loaded, sess
	if not model_loaded:
		raise RuntimeError("ERROR: must init forward model first")
	processed_img = image_preprocessor(input_image)
	p = sess.run(probs, feed_dict={image: processed_img})[0]
	return p

def get_logits_probs_image_tf(sess):
	global logits, probs, image, model_loaded
	if not model_loaded:
		init(sess)
	return logits, probs, image




