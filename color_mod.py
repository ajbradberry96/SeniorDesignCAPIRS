import PIL
import PIL.Image
import PIL.ImageEnhance
import numpy as np
import random


def color_mod(img):
	new_img = blueify(saturate_mod(img))
	return new_img

def saturate_mod(img):
	converter = PIL.ImageEnhance.Color(img)
	#saturate_range = random.uniform(0.5, 2)
	img2 = converter.enhance(3)
	return img2

def blueify(img):
	arr = np.asarray(img).copy()
	for i in range(arr.shape[0]):
		for j in range(arr.shape[1]):
			new_num = arr[i,j,2]+30
			if new_num >= 255:
				new_num = 255
			arr[i,j,2] = new_num
	return PIL.Image.fromarray(np.uint8(arr))
