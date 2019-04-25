import tensorflow as tf
import pandas as pd

from testing import forward_model
from testing import detect_adversarial
from testing import image_processing
import matplotlib
import seaborn
from matplotlib import pyplot as plt

from urllib.request import urlretrieve

import PIL
import os

tf.logging.set_verbosity(tf.logging.ERROR)
sess = tf.InteractiveSession()

img_path = "media/dataset/"

#forward_model.init(sess)

#file_arr = os.listdir(img_path)
#pil_imgs = [PIL.Image.open(img_path+x) for x in file_arr]

#detect_adversarial.detect_test(pil_imgs,file_arr,separate_advs=False)


distort_ops = ["warp","colorshift","saturate","noise","average"]


dist_dict = {x:[] for x in distort_ops}
for op in distort_ops:
	dist_dict["adv"+op] = []

with open("cosine_data.csv","r") as f:
	header = f.readline()
	for line in f:
		line_parts = line.strip().split(",")
		for op in distort_ops:
			if op in line_parts[0]:
				if "adv" in line_parts[0]:
					dist_dict["adv"+op].append(float(line_parts[1]))
				else:
					dist_dict[op].append(float(line_parts[1]))

xoff=0
bar_xs = []
bar_heights = []
bar_names = []
for op in distort_ops:
	bar_xs.append(xoff)
	bar_heights.append(sum(dist_dict[op])/len(dist_dict[op]))
	bar_names.append(op)

	xoff+=0.5
	bar_xs.append(xoff)
	bar_heights.append(sum(dist_dict["adv"+op])/len(dist_dict["adv"+op]))
	bar_names.append("adv " + op)
	xoff+=1

plt.figure()
plt.bar(bar_xs, bar_heights, 0.4, color=["b","r"]*len(distort_ops), tick_label=bar_names)

val_sets = []
for op in distort_ops:
	val_sets.append(dist_dict[op])
	val_sets.append(dist_dict["adv" + op])

plt.figure()
plt.boxplot(val_sets, labels=bar_names)

for i in range(len(val_sets)//2):
	plt.figure()
	plt.title(bar_names[i*2])
	seaborn.kdeplot(val_sets[i*2],color="b")
	print(distort_ops[i])
	print(val_sets[i*2+1])
	seaborn.kdeplot(val_sets[i*2+1],color="r")
plt.show()

from sklearn.linear_model import LogisticRegression

#distort_ops = ["warp","colorshift","saturate","noise","average"]
distort_ops = ["warp","colorshift","saturate","noise"]
#distort_ops = ["average"]
X_data_dict_norm = {}
X_data_dict_adv = {}

with open("cosine_data.csv","r") as f:
	header = f.readline()
	for line in f:
		line_parts = line.strip().split(",")
		for i, op in enumerate(distort_ops):
			if op in line_parts[0]:
				use_dict = X_data_dict_adv if "adv" in line_parts[0] else X_data_dict_norm
				base_name = line_parts[0].split("_" + op)[0]
				if base_name not in use_dict:
					use_dict[base_name] = [0] * len(distort_ops)
				use_dict[base_name][i] += float(line_parts[1]) * (1/3)

X_data = []
Y_data = []
for entry in X_data_dict_norm:
	X_data.append(X_data_dict_norm[entry])
	Y_data.append(0)

for entry in X_data_dict_adv:
	X_data.append(X_data_dict_adv[entry])
	Y_data.append(1)

lr = LogisticRegression()
lr.fit(X_data, Y_data)

predictions = []
for entry in X_data:
	predictions.append(lr.predict_proba([entry])[0][1])

print(predictions)

#AVG_data = [x[0] for x in X_data]
#dat_frame = pd.DataFrame({"X":AVG_data,"Y":Y_data})
#plt.figure()
#seaborn.regplot("X","Y",dat_frame,logistic=True)


import sklearn.metrics as metrics
fpr, tpr, threshold = metrics.roc_curve(Y_data, predictions)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.title('Receiver Operating Characteristic (Multiple)')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


LogisticRegression().fit

"""
Questions to ask
Which distortions are the best discriminators?
AUC curve
PCA of images"""
