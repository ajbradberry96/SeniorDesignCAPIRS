from scipy import spatial
import numpy as np
import forward_model
import image_processing


def detect(img):
	"""
	Detects an adversarial example if one exists

	Takes in a PIL image. Returns True if the image is an adversarial example
	"""

	orig_vector = list(forward_model.predict(img))

	transform_vectors = []

	for i in range(3):
		col_img = image_processing.color_shift(img)
		t_vec = list(forward_model.predict(col_img))
		transform_vectors.append(t_vec)

	for i in range(3):
		sat_img = image_processing.saturate_mod(img)
		t_vec = list(forward_model.predict(sat_img))
		transform_vectors.append(t_vec)

	for i in range(3):
		noise_img = image_processing.add_noise(img)
		t_vec = list(forward_model.predict(noise_img))
		transform_vectors.append(t_vec)

	for i in range(3):
		warp_img = image_processing.rand_warp(img)
		t_vec = list(forward_model.predict(warp_img))
		transform_vectors.append(t_vec)

	average_trans_vector = list(np.average(transform_vectors, axis=0))
	cosine_diff = spatial.distance.cosine(orig_vector, average_trans_vector)

	print(cosine_diff)

	if cosine_diff > 0.01:
		return True
	else:
		return False
