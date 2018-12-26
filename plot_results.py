import matplotlib.pyplot as plt
import forward_model

def plot(img, probs):
	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
	fig.sca(ax1)
	ax1.imshow(img)
	fig.sca(ax1)

	imagenet_labels = forward_model.get_imagenet_labels()
	topk = list(probs.argsort()[-10:][::-1])
	topprobs = probs[topk]
	barlist = ax2.bar(range(10), topprobs)
	plt.sca(ax2)
	plt.ylim([0, 1.1])
	plt.xticks(range(10),
			   [imagenet_labels[i][:15] for i in topk],
			   rotation='vertical')
	fig.subplots_adjust(bottom=0.2)
	plt.show()
