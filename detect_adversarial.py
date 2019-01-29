from scipy import spatial
import numpy as np
import forward_model
import image_processing
import os


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

def detect_test(imgs, names, separate_advs=True):
	"""
	Generate data for our algorithms

	Takes in a list of PIL images (imgs) and image names (names)
	separate_advs determines whether or not image names should be used to split adversarial exmples from true examples for easier analysis
		If true, then search if the name contains "adversarial"
			If it does, then put it into the vector_data_adv.csv file and cosine_data_adv.csv file
			Otherwise, put it into the vector_data_true.csv file and the cosine_data_true.csv file
			E.g. cat_image_adversarial would go into the adversarial data
				cat_image_normal would go into the true data
		If separate_advs is False, then log into vector_data and cosine_data files
	Generates data of how our classifiers performed.
	Outputs this data as a list for every image.
	"""

	if separate_advs:
		with open("vector_data_true.csv","w") as f:
			# Write out the headers for the vector data
			f.write("name,"+",".join(forward_model.get_imagenet_labels()) + "\n")

		with open("cosine_data_true.csv","w") as f:
			# Write out the headers for the simple cosine difference data
			f.write("name,"+",".join(forward_model.get_imagenet_labels()) + "\n")

		with open("vector_data_adv.csv","w") as f:
			# Write out the headers for the vector data
			f.write("name,"+",".join(forward_model.get_imagenet_labels()) + "\n")

		with open("cosine_data_adv.csv","w") as f:
			# Write out the headers for the simple cosine difference data
			f.write("name,"+",".join(forward_model.get_imagenet_labels()) + "\n")
	else:
		with open("vector_data.csv","w") as f:
			# Write out the headers for the vector data
			f.write("name,"+",".join(forward_model.get_imagenet_labels()) + "\n")

		with open("cosine_data.csv","w") as f:
			# Write out the headers for the simple cosine difference data
			f.write("name,"+",".join(forward_model.get_imagenet_labels()) + "\n")

	for i, img in enumerate(imgs):
		img_name = names[i]

		if separate_advs:
			if "adversarial" in img_name:
				vec_file_name = "vector_data_adv.csv"
				cos_file_name = "cosine_data_adv.csv"
			else:
				vec_file_name = "vector_data_true.csv"
				cos_file_name = "cosine_data_true.csv"
		else:
			vec_file_name = "vector_data.csv"
			cos_file_name = "cosine_data.csv"


		orig_vector = list(forward_model.predict(img))
		vec_name = "_orig"
		with open(vec_file_name) as f:
			f.write(img_name+vec_name+","+ ",".join([str(x) for x in orig_vector]) + "\n")

		transform_vectors = []

		for i in range(3):
			col_img = image_processing.color_shift(img)
			t_vec = list(forward_model.predict(col_img))
			transform_vectors.append(t_vec)

			vec_name = "_colorshift" + str(i)
			with open(vec_file_name, "a") as f:
				f.write(img_name+vec_name+","+ ",".join([str(x) for x in t_vec]) + "\n")

			cosine_diff = spatial.distance.cosine(orig_vector, t_vec)
			with open(cos_file_name, "a") as f:
				f.write(img_name+vec_name+"," + str(cosine_diff) + "\n")

		for i in range(3):
			sat_img = image_processing.saturate_mod(img)
			t_vec = list(forward_model.predict(sat_img))
			transform_vectors.append(t_vec)

			vec_name = "_saturate" + str(i)
			with open(vec_file_name, "a") as f:
				f.write(img_name+vec_name+","+ ",".join([str(x) for x in t_vec]) + "\n")

			cosine_diff = spatial.distance.cosine(orig_vector, t_vec)
			with open(cos_file_name, "a") as f:
				f.write(img_name+vec_name+"," + str(cosine_diff) + "\n")

		for i in range(3):
			noise_img = image_processing.add_noise(img)
			t_vec = list(forward_model.predict(noise_img))
			transform_vectors.append(t_vec)

			vec_name = "_noise" + str(i)
			with open(vec_file_name, "a") as f:
				f.write(img_name+vec_name+","+ ",".join([str(x) for x in t_vec]) + "\n")

			cosine_diff = spatial.distance.cosine(orig_vector, t_vec)
			with open(cos_file_name, "a") as f:
				f.write(img_name+vec_name+"," + str(cosine_diff) + "\n")

		for i in range(3):
			warp_img = image_processing.rand_warp(img)
			t_vec = list(forward_model.predict(warp_img))
			transform_vectors.append(t_vec)

			vec_name = "_warp" + str(i)
			with open(vec_file_name, "a") as f:
				f.write(img_name+vec_name+","+ ",".join([str(x) for x in t_vec]) + "\n")

			cosine_diff = spatial.distance.cosine(orig_vector, t_vec)
			with open(cos_file_name, "a") as f:
				f.write(img_name+vec_name+"," + str(cosine_diff) + "\n")


		average_trans_vector = list(np.average(transform_vectors, axis=0))
		cosine_diff = spatial.distance.cosine(orig_vector, average_trans_vector)

		vec_name = "_average"
		with open(vec_file_name, "a") as f:
			f.write(img_name+vec_name+","+",".join([str(x) for x in average_trans_vector]) + "\n")

		with open(cos_file_name, "a") as f:
			f.write(img_name+vec_name+"," + str(cosine_diff) + "\n")




