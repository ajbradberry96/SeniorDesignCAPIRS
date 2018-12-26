import tensorflow as tf
import numpy as np
import PIL
import forward_model

def generate_adversarial_example(img, sess, mode="normal"):
	logits, probs, image = forward_model.get_logits_probs_image_tf(sess)
	img = forward_model.image_preprocessor(img)

	x = tf.placeholder(tf.float32, (299, 299, 3))

	x_hat = image # our trainable adversarial input
	assign_op = tf.assign(x_hat, x)

	learning_rate = tf.placeholder(tf.float32, ())
	y_hat = tf.placeholder(tf.int32, ())

	labels = tf.one_hot(y_hat, 1000)

	epsilon = tf.placeholder(tf.float32, ())

	below = x - epsilon
	above = x + epsilon
	projected = tf.clip_by_value(tf.clip_by_value(x_hat, below, above), 0, 1)
	with tf.control_dependencies([projected]):
		project_step = tf.assign(x_hat, projected)


	if mode=="normal":
		loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=[labels])
		optim_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, var_list=[x_hat])

		demo_epsilon = 2.0/255.0 # a really small perturbation
		demo_lr = 1e-1
		demo_steps = 100
		demo_target = 924 # "guacamole"

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
		demo_target = 924  # "guacamole"

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

	adv = x_hat.eval() # retrieve the adversarial example

	adv_img = PIL.Image.fromarray(np.uint8((adv)*255))

	return adv_img
