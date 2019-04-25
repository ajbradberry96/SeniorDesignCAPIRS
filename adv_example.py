import tensorflow as tf
import numpy as np
import PIL
import forward_model

def pil_paste(original,paste_img):
	print(original)
	img_input = np.array(original).astype('uint8')
	img = PIL.Image.fromarray(img_input)

	paste_img_input = np.array(paste_img).astype('uint8')
	paste_img = PIL.Image.fromarray(paste_img_input)

	img.paste(paste_img, (40, 40))

	ret_val = np.asarray(img)

	return ret_val


def generate_adversarial_example(img, sess, mode="normal", adv_class="guacamole"):
	logits, probs, image = forward_model.get_logits_probs_image_tf(sess)
	img = forward_model.image_preprocessor(img)

	x = tf.placeholder(tf.float32, (299, 299, 3))

	x_hat = image # our trainable adversarial input
	assign_op = tf.assign(x_hat, x)

	learning_rate = tf.placeholder(tf.float32, ())
	y_hat = tf.placeholder(tf.int32, ())

	labels = tf.one_hot(y_hat, 1000)



	if mode=="normal":
		epsilon = tf.placeholder(tf.float32, ())
		below = x - epsilon
		above = x + epsilon
		projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
		with tf.control_dependencies([projected]):
			project_step = tf.assign(x_hat, projected)

		loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
		optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])

		demo_epsilon = 2.0/255.0 # a really small perturbation
		demo_lr = 1e-1
		demo_steps = 100
		labels = forward_model.get_imagenet_labels()
		demo_target = labels.index(adv_class)

		# initialization step
		sess.run(assign_op, feed_dict={x: img})

		# projected gradient descent
		for i in range(demo_steps):
			# gradient descent step
			_, loss_value = sess.run(
				[optim_step, loss],
				feed_dict={learning_rate: demo_lr, y_hat: demo_target})
			# project step
			sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
			if (i+1) % 10 == 0:
				print('step %d, loss=%g' % (i+1, loss_value))

	elif mode=="rot_robust":
		epsilon = tf.placeholder(tf.float32, ())
		below = x - epsilon
		above = x + epsilon
		projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
		with tf.control_dependencies([projected]):
			project_step = tf.assign(x_hat, projected)

		num_samples = 10
		average_loss = 0
		for i in range(num_samples):
			rotated = tf.contrib.image.rotate(
				image, tf.random_uniform((), minval=-np.pi/4, maxval=np.pi/4))
			rotated_logits, _ = forward_model.inception(rotated, reuse=True)
			average_loss += tf.nn.softmax_cross_entropy_with_logits(
				logits=rotated_logits, labels=labels) / num_samples

		optim_step = tf.train.GradientDescentOptimizer(
			learning_rate).minimize(average_loss, var_list=[x_hat])

		demo_epsilon = 8.0 / 255.0  # still a pretty small perturbation
		demo_lr = 2e-1
		demo_steps = 100
		labels = forward_model.get_imagenet_labels()
		demo_target = labels.index(adv_class)

		# initialization step
		sess.run(assign_op, feed_dict={x: img})

		# projected gradient descent
		for i in range(demo_steps):
			# gradient descent step
			_, loss_value = sess.run(
				[optim_step, average_loss],
				feed_dict={learning_rate: demo_lr, y_hat: demo_target})
			# project step
			sess.run(project_step, feed_dict={x: img, epsilon: demo_epsilon})
			if (i + 1) % 50 == 0:
				print('step %d, loss=%g' % (i + 1, loss_value))


	elif mode=="sticker1":

		sticker = tf.Variable(tf.zeros((29, 29, 3)))

		num_samples = 1
		average_loss = 0
		for i in range(num_samples):
			x_hat_paste = tf.assign(x_hat[50:79,50:79,:],sticker)
			pasted_logits, _ = forward_model.inception(x_hat_paste, reuse=True)
			average_loss += tf.nn.softmax_cross_entropy_with_logits(
				logits=pasted_logits, labels=labels) / num_samples

		optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(average_loss, var_list=[sticker])

		demo_epsilon = 2.0/255.0 # a really small perturbation
		demo_lr = 1e-1
		demo_steps = 100
		labels = forward_model.get_imagenet_labels()
		demo_target = labels.index(adv_class)

		# initialization step
		pil_img = PIL.Image.new('RGB', (29, 29), color='black')
		black_box = (np.asarray(pil_img) / 255.0).astype(np.float32)
		sticker_init = tf.assign(sticker,x)
		sess.run(assign_op, feed_dict={x: black_box})

		# projected gradient descent
		for i in range(demo_steps):
			# gradient descent step
			_, loss_value = sess.run(
				[optim_step, average_loss],
				feed_dict={learning_rate: demo_lr, y_hat: demo_target})
			# project step
			if (i+1) % 10 == 0:
				print('step %d, loss=%g' % (i+1, loss_value))

	elif mode=="sticker":
		epsilon = tf.placeholder(tf.float32, ())
		below = x - epsilon
		above = x + epsilon
		projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
		with tf.control_dependencies([projected]):
			project_step = tf.assign(x_hat, projected)

		x_blank = tf.Variable(tf.zeros((299, 299, 3)))

		x_blank_paste_full = tf.assign(x_blank,x)
		#x_blank_paste_partial = tf.assign(x_blank[50:150,50:150,:],x_hat[50:150,50:150,:])
		x_blank_paste_partial = tf.assign(x_blank,x_hat)
		x_hat_paste_full = tf.assign(x_hat,x_blank)

		loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
		optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])

		demo_epsilon = 2.0/255.0 # a really small perturbation
		demo_lr = 1e-1
		demo_steps = 100
		labels = forward_model.get_imagenet_labels()
		demo_target = labels.index(adv_class)

		# initialization step
		pil_img = PIL.Image.new('RGB', (299, 299), color='black')
		black_box = (np.asarray(pil_img) / 255.0).astype(np.float32)
		sess.run(x_blank_paste_full, feed_dict={x: img})
		sess.run(assign_op, feed_dict={x:black_box})
		sess.run(x_blank_paste_partial)
		sess.run(x_hat_paste_full)
		#sess.run(assign_op, feed_dict={x: img})

		# projected gradient descent
		for i in range(demo_steps):
			# gradient descent step
			_, loss_value = sess.run(
				[optim_step, loss],
				feed_dict={learning_rate: demo_lr, y_hat: demo_target})
			# project step
			sess.run(x_blank_paste_full, feed_dict={x: img})
			partial_img = sess.run(x_blank_paste_partial)
			sess.run(assign_op, feed_dict={x:partial_img})
			if (i+1) % 10 == 0:
				print('step %d, loss=%g' % (i+1, loss_value))


	adv = x_hat.eval()  # retrieve the adversarial example

	adv_img = PIL.Image.fromarray(np.uint8((adv)*255))

	return adv_img

if __name__ == "__main__":

	tf.logging.set_verbosity(tf.logging.ERROR)
	sess = tf.InteractiveSession()

	img = PIL.Image.open("media/stop_sign_.jpg")
	img = img.resize((300, 300), PIL.Image.ANTIALIAS)
	img.save("media/norm_img.png")

	forward_model.init(sess)

	print(forward_model.get_imagenet_labels())

	adv_img = generate_adversarial_example(img, sess, adv_class='milk can')

	adv_img.save("media/adv_img.png")

