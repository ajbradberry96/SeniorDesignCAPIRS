import matplotlib.pyplot as plt
import forward_model


def plot(img, probs, cos_diff=-1):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
	fig.sca(ax1)
	ax1.imshow(img)
	fig.sca(ax1)

	imagenet_labels = forward_model.get_imagenet_labels()
	topk = list(probs.argsort()[-10:][::-1])
	topprobs = probs[topk]
	barlist = ax2.bar(range(10), topprobs)
	if cos_diff != -1:
		plt.text(0,-100,"Cosine Distance: " + str(round(cos_diff,3)))
	plt.sca(ax2)
	plt.ylim([0, 1.1])
	plt.xticks(range(10),
			[imagenet_labels[i][:15] for i in topk],
			rotation='vertical')
	fig.subplots_adjust(bottom=0.2)

if __name__ == "__main__":
	import tensorflow as tf
	import PIL
	tf.logging.set_verbosity(tf.logging.ERROR)
	sess = tf.InteractiveSession()

	#img = PIL.Image.open("media/norm_img.png")
	#adv_img = PIL.Image.open("media/adv_img.png")

	img = PIL.Image.open("media/pistol.png")
	adv_img = PIL.Image.open("media/pistol_adv.png")

	forward_model.init(sess)

	norm_probs = forward_model.predict(img)
	adv_probs = forward_model.predict(adv_img)

	plot(img,norm_probs)
	plot(adv_img,adv_probs)
	plt.show()

