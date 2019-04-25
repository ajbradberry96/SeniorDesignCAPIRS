from scipy import spatial
import numpy as np
from capirs import image_processing
import os


threshold = 0.01


def predict_default(img):
	# MODIFY THIS FUNCTION TO INTERACT WITH YOUR FORWARD MODEL
	# Feed in an image, return a list of probabilities
	return []

predict = predict_default


def detect(img):
	"""
	Detects an adversarial example if one exists

	Takes in a PIL image. Returns True if the image is an adversarial example
	"""

	orig_vector = list(predict(img))

	transform_vectors = []

	for i in range(3):
		col_img = image_processing.color_shift(img)
		t_vec = predict(col_img)
		transform_vectors.append(list(t_vec))
		cosine_diff = spatial.distance.cosine(orig_vector, t_vec)

	for i in range(3):
		sat_img = image_processing.saturate_mod(img)
		t_vec = predict(sat_img)
		transform_vectors.append(list(t_vec))
		cosine_diff = spatial.distance.cosine(orig_vector, t_vec)

	for i in range(3):
		noise_img = image_processing.add_noise(img)
		t_vec = predict(noise_img)
		transform_vectors.append(list(t_vec))
		cosine_diff = spatial.distance.cosine(orig_vector, t_vec)

	for i in range(3):
		warp_img = image_processing.rand_warp(img)
		t_vec = predict(warp_img)
		transform_vectors.append(list(t_vec))
		cosine_diff = spatial.distance.cosine(orig_vector, t_vec)

	average_trans_vector = list(np.average(transform_vectors, axis=0))
	cosine_diff = spatial.distance.cosine(orig_vector, average_trans_vector)

	if cosine_diff > threshold:
		return True
	else:
		return False
