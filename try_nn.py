# -*- coding: utf-8 -*-
"""
Created on Thu Sep 22 20:20:08 2016
@author: kaavee
python try_nn.py 100 1 0.001
"""

import tensorflow as tf
from skimage import data, io, data_dir, transform, viewer, morphology
import numpy as np
import random
import sys
from sklearn import preprocessing

np.set_printoptions(threshold=np.nan)

def step_function(x):
	if x>0:
		return 1
	else:
		return 0


# print("0")


cleandata = np.load('train_data.npy')

to_take = cleandata.shape[1]-2
# to_take = cleandata.shape[1]-1

labels = list(range(0,35,1)) + list(range(36, cleandata.shape[1]-1, 1))
train_x=cleandata[:,labels]
scaler = preprocessing.StandardScaler()
train_x = scaler.fit_transform(train_x)
train_y = cleandata[:,[35]]

valid_data = np.load('data.npy')

valid_x = valid_data[:,labels]
valid_x = scaler.transform(valid_x)
valid_y = valid_data[:,[35]]

# print("1")

no_inp = to_take

x = tf.placeholder(tf.float32, [None, no_inp])

hl_1=int(sys.argv[1])
hl_2=int(sys.argv[2])
learning_rate = float(sys.argv[3])
out_l=1

W0 = tf.Variable(tf.random_normal([no_inp, hl_1],mean=0.00, stddev=0.0001))
# W0 = tf.Print(W0, [W0], message="This is W0: ", summarize = 10)
b0 = tf.Variable(tf.random_normal([hl_1],mean=0.00, stddev=0.0001))
# b0 = tf.Print(b0, [b0], message="This is b0: ", summarize = 10)
z0 = tf.matmul(x, W0) + b0
# z0 = tf.Print(z0, [z0], message="This is z0: ", summarize = 10)
h0=tf.nn.relu(z0)
# h0 = tf.Print(h0, [h0], message="This is h0: ", summarize = 104)


W1 = tf.Variable(tf.random_normal([hl_1, hl_2],mean=0.00, stddev=0.0001))
# W1 = tf.Print(W1, [W1], message="This is W1: ", summarize = 10)
b1 = tf.Variable(tf.random_normal([hl_2],mean=0.00, stddev=0.0001))
# b1 = tf.Print(b1, [b1], message="This is b1: ", summarize = 10)
z1 = tf.matmul(h0, W1) + b1
# z1 = tf.Print(z1, [z1], message="This is z1: ", summarize = 10)
h1=tf.nn.relu(z1)
# y=tf.nn.sigmoid(z1)
# y = tf.Print(y, [y], message="This is y: ", summarize = 10)

W2 = tf.Variable(tf.random_normal([hl_2, out_l],mean=0.00, stddev=0.0001))
# W2 = tf.Print(W2, [W2], message="This is W2: ", summarize = 10)
b2 = tf.Variable(tf.random_normal([out_l],mean=0.00, stddev=0.0001))
# b2 = tf.Print(b2, [b2], message="This is b2: ", summarize = 10)
z2 = tf.matmul(h1, W2) + b2
# z2 = tf.Print(z2, [z2], message="This is z2: ", summarize = 10)
y=tf.nn.sigmoid(z2)
y=z2


# # y_reduce = tf.reduce_sum(y,1)
# # y_reduce = tf.Print(y_reduce, [y_reduce], message="This is y_reduce: ", summarize = 100000)



y_ = tf.placeholder(tf.float32, [None, out_l])
# y_ = tf.Print(y_, [y_], message="This is y_real: ", summarize = 10)

# cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
cross_entropy = -tf.reduce_sum(y_*tf.log(tf.clip_by_value(y,1e-10,1.0)) + (1-y_)*tf.log(tf.clip_by_value(1-y,1e-10,1.0)))
# cross_entropy = tf.reduce_sum((y_ - y)*(y_ - y))
# cross_entropy = tf.Print(cross_entropy,[cross_entropy],"This is cross entropy: ")

# reg0 = 0.01*tf.nn.l2_loss(W0) + 0.01*tf.nn.l2_loss(b0)
reg0 = 0
# reg1 = 0.01*tf.nn.l2_loss(W1) + 0.01*tf.nn.l2_loss(b1)
reg1 = 0
# reg2 = 0.01*tf.nn.l2_loss(W2) + 0.01*tf.nn.l2_loss(b2)
reg2 = 0

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy + reg2 + reg1 + reg0)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

print("start")

iterations1=10000


for i in range(iterations1):
	if((i%100)==0):
		print(i)
	sample_size=2000	
	batch_xs = np.zeros((sample_size,no_inp))
	batch_ys =np.zeros((sample_size,out_l))
	indices = random.sample(range(train_x.shape[0]),sample_size)
	for j in range(sample_size):
		a=indices[j]
		batch_xs[j]=train_x[a]
		batch_ys[j]=train_y[a]
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	if((i%100)==0):
		print(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
		nd_y = sess.run(y,feed_dict={x:valid_x, y_:valid_y})
		# print nd_y.shape
		# for i in [0, 10, 1000, 10000, 100000]:
		# 	print(nd_y[i])

		vstep = np.vectorize(step_function)
		y_mean = np.mean(nd_y)
		# print('yoyoyo - ',y_mean)
		nd_y2 = vstep(nd_y - y_mean)
		valid_y2 = vstep(valid_y)
		count1=0
		count2=0
		accuracy=0
		true_positive=0
		true_negitive=0
		false_positive=0
		false_negitive=0
		for i in range(valid_y2.shape[0]):
			if(i>0 and valid_x[i][0]!=valid_x[i-1][0]):
				if(count2/count1 > 0.00 and valid_y2[i-1]==1):
					true_positive+=1
				if(count2/count1 > 0.00 and valid_y2[i-1]==0):
					false_positive+=1
				if(count2/count1 <= 0.00 and valid_y2[i-1]==0):
					true_negitive+=1
				if(count2/count1 <= 0.00 and valid_y2[i-1]==1):
					false_negitive+=1
				count1=1
				count2=nd_y2[i]
			else:
				count1+=1
				count2+=nd_y2[i]

		print("accuracy - ",np.equal(valid_y2,nd_y2).mean())
		# false_positive = 0
		# true_positive = 0
		# false_negitive = 0
		# true_negitive = 0
		# for i in range(valid_y2.shape[0]):
		# 	if(nd_y2[i]==1 and valid_y2[i]==1):
		# 		true_positive+=1
		# 	if(nd_y2[i]==1 and valid_y2[i]==0):
		# 		false_positive+=1
		# 	if(nd_y2[i]==0 and valid_y2[i]==1):
		# 		false_negitive+=1
		# 	if(nd_y2[i]==0 and valid_y2[i]==0):
		# 		true_negitive+=1
		sum = (false_positive+false_negitive+true_negitive+true_positive)*1.0

		print("values - ",true_positive/sum, true_negitive/sum, false_positive/sum, false_negitive/sum)
nd_y = sess.run(y,feed_dict={x:train_x, y_:train_y})
np.save('results',nd_y)

sess.close()