#encoding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import argparse
import sys
import os
from pre_process import load_data, get_batch, get_each_query_length_for_model
import tensorflow as tf 
import numpy as np 
import time
# from numpy.random import seed
# seed(1)
# from tensorflow import set_random_seed
# set_random_seed(2)
# rng = np.random.RandomState(23455)

def set_config():
	os.environ['CUDA_VISIBLE_DEVICES'] = '0'    
	gpu_options = tf.GPUOptions(allow_growth=True)
	config = tf.ConfigProto(gpu_options=gpu_options)
	return config

class QRL_L2R(object):
	def __init__(self, feature_dim, learn_rate, reward_decay, FLAGS):
		
		self.feature_dim = feature_dim
		self.lr = -learn_rate
		self.FLAGS = FLAGS
		self.stdv = 0.5
		# self.current_step = 0
		self.gamma = reward_decay
		self.layer1_node_num = 128

		self.layer2_node_num = 128

		self.layer3_node_num = 64

		self.layer4_node_num = 32
		# self.sess = tf.Session()
		self._build_net()		
		
		self.sess = tf.Session(config=set_config())
		# self.sess.as_default()
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep = 4)			
		# self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())
		# self.current_step = 0
		self.ep_docs, self.ep_label, self.ep_rs = [],[],[]		
		# self.reduced_matrix = tf.eye(self.feature_dim)
		# self.tvars = tf.trainable_variables()

	def _build_net(self):
		with tf.name_scope("inputs"):
			self.input_doc_feature = tf.placeholder(tf.float32,[None, self.feature_dim],name = "doc_feature_list")
			self.idel_doc_feature = tf.placeholder(tf.float32,[None, self.feature_dim],name = "idel_feature")
			self.reduced_matrix = tf.placeholder(tf.float32,[self.feature_dim, self.feature_dim],name = "reduced_matrix")
			self.tf_rt = tf.placeholder(tf.float32,[None,1],name = "reward_value")
			# self.input_y = tf.placeholder(tf.float32,[None,1],name = "ep_label")			
			self.input_y = tf.placeholder(tf.float32,[None,],name = "ep_label")

			self.W1Grad = tf.placeholder(tf.float32, [self.feature_dim, self.layer1_node_num],name='w1_grad')
			self.b1Grad = tf.placeholder(tf.float32, [self.layer1_node_num,],name='b1_grad')
			self.W2Grad = tf.placeholder(tf.float32, [self.layer1_node_num, self.layer2_node_num],name='w2_grad')
			self.b2Grad = tf.placeholder(tf.float32, [self.layer2_node_num,],name='b2_grad')

			self.W3Grad = tf.placeholder(tf.float32, [self.layer2_node_num, self.layer3_node_num],name='w3_grad')
			self.b3Grad = tf.placeholder(tf.float32, [self.layer3_node_num,],name='b3_grad')
			self.W4Grad = tf.placeholder(tf.float32, [self.layer3_node_num, self.layer4_node_num],name='w4_grad')
			self.b4Grad = tf.placeholder(tf.float32, [self.layer4_node_num,],name='b4_grad')

			self.W5Grad = tf.placeholder(tf.float32, [self.layer4_node_num, self.feature_dim],name='w5_grad')
			self.b5Grad = tf.placeholder(tf.float32, [self.feature_dim,],name='b5_grad')
			
			self.dropout_keep_prob = tf.placeholder(tf.float32, name = "dropout_keep_prob")
			# tvars = tf.trainable_variables()			
			# self.tvars = tf.trainable_variables()
		# ranking process

		# all_act = tf.matmul(self.input_doc_feature, self.reduced_matrix)
		all_act = self.param_network(tf.matmul(self.input_doc_feature, self.reduced_matrix), self.dropout_keep_prob)

		# print ("all_act shape : {}".format(all_act.get_shape()))

		scores = tf.diag_part(tf.matmul(all_act, tf.transpose(self.input_doc_feature, perm=[1,0])))

		# print ("scores shape : {}".format(scores.get_shape()))

		prob = tf.nn.softmax(scores, name='prob')			
		# prob = tf.expand_dims(prob,1)
		# print ("prob shape : {}".format(prob.get_shape()))
		
		# get next document index
		self.d_index = tf.argmax(scores)
		
		with tf.name_scope("train"):			
			self.tvars = tf.trainable_variables()
			# loss 
			likelihood = self.input_y*(self.input_y - scores) + (1 - self.input_y)*(self.input_y + scores)
			log_like = tf.log(tf.clip_by_value(likelihood,1e-10,1.0))
			self.loss = -tf.reduce_mean(log_like * self.tf_rt)
			
			# p_label = tf.nn.softmax(self.input_y, dim = 0)
			# self.loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels = p_label, logits=scores, dim = 0))
						
			# self.loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=prob, labels=self.input_y)

			# likelihood = self.input_y*(self.input_y - scores) + (1 - self.input_y)*(self.input_y + scores)
			# log_like = tf.log(tf.clip_by_value(likelihood,1e-10,1.0))
			# self.loss = -tf.reduce_mean(log_like * self.tf_rt)
			# update
			self.newGrads = tf.gradients(self.loss, self.tvars)	
			batchGrad = [self.W1Grad, self.b1Grad, self.W2Grad, self.b2Grad,self.W3Grad, self.b3Grad,self.W4Grad, self.b4Grad,self.W5Grad, self.b5Grad]

			adam = tf.train.AdamOptimizer(learning_rate = self.lr)
			self.train_op = adam.apply_gradients(zip(batchGrad, self.tvars))

			# SGD = tf.train.GradientDescentOptimizer(self.lr)
			# self.train_op = SGD.apply_gradients(zip(batchGrad, self.tvars))
			
	def param_network(self, net_input,dropout_rate):

		# tf.reset_default_graph()
		# print ("net_input shape : {}".format(net_input.get_shape()))
		layer1 = tf.layers.dense(
					inputs=net_input,
					units=self.layer1_node_num,
					activation=tf.nn.tanh,
					kernel_initializer=tf.random_normal_initializer(mean = 0, stddev= 0.1 / np.sqrt(float(self.feature_dim))),
					bias_initializer=tf.constant_initializer(0.1),
					name = "layer1")

		# print ("layer1 shape : {}".format(layer1.get_shape()))
		layer2 = tf.layers.dense(
			inputs=layer1,
			units=self.layer2_node_num,
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean = 0, stddev=0.1 / np.sqrt(float(self.layer1_node_num))),
			bias_initializer=tf.constant_initializer(0.1),
			name = "layer2")

		layer3 = tf.layers.dense(
			inputs=layer2,
			units=self.layer3_node_num,
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean = 0, stddev=0.1 / np.sqrt(float(self.layer2_node_num))),
			bias_initializer=tf.constant_initializer(0.1),
			name = "layer3")

		layer4 = tf.layers.dense(
			inputs=layer3,
			units=self.layer4_node_num,
			activation=tf.nn.tanh,
			kernel_initializer=tf.random_normal_initializer(mean = 0, stddev=0.1 / np.sqrt(float(self.layer3_node_num))),
			bias_initializer=tf.constant_initializer(0.1),
			name = "layer4")

		drop_layer = tf.layers.dropout(layer4, dropout_rate)

		output = tf.layers.dense(
			# inputs=layer1,
			inputs=drop_layer,
			units=self.feature_dim,
			activation=None,
			kernel_initializer=tf.random_normal_initializer(mean = 0, stddev=0.1 / np.sqrt(float(self.layer4_node_num))),
			bias_initializer=tf.constant_initializer(0.1),
			name = "out_layer")
		# print ("out_layer shape:{}".format(output.get_shape()))
		return output

	def calcu_reduced_matrix(self,remain_doc_feature):

		# calcu t-th reduced matrix
		
		# ep_doc1 = np.array(self.ep_docs[len(self.ep_docs)-2])[:,np.newaxis]
		# ep_doc2 = np.array(self.ep_docs[len(self.ep_docs)-1])[:,np.newaxis]

		ep_doc1 = np.array(self.ep_docs[len(self.ep_docs)-2]).reshape((self.feature_dim, 1))		
		ep_doc2 = np.array(self.ep_docs[len(self.ep_docs)-1]).reshape((self.feature_dim, 1))
		# print ("ep_doc1 : shape : {}".format(ep_doc1.shape))
		# print ("ep_doc2 : shape : {}".format(ep_doc2.shape))
		# exit()

		# |1><2|
		out_product_1_2 = np.matmul(ep_doc1, ep_doc2.T) 
		# |1><1|
		out_product_1_1 = np.matmul(ep_doc1, ep_doc1.T)  
		# |2><1|
		out_product_2_1 = np.matmul(ep_doc2, ep_doc1.T) 
		# |2><2|
		out_product_2_2 = np.matmul(ep_doc2, ep_doc2.T)

		out_product = out_product_1_1 + out_product_1_2 + out_product_2_1 + out_product_2_2

		inner_product = np.matmul(remain_doc_feature, remain_doc_feature.T)
		inner_product = np.sum(np.reshape(inner_product,(inner_product.size,)))
		reduced_matrix = out_product * inner_product

		return_matrix = self.matrix_standardized(reduced_matrix)
		# print ("return_matrix shape : {}".format(return_matrix.shape))
		# exit()
		return return_matrix

	def matrix_standardized(self, matrix):
		
		standard_matrix = matrix
		standard_matrix -= np.mean(matrix)
		standard_matrix /= np.std(matrix)

		return standard_matrix
	

	def choose_doc(self, current_step, remain_doc_feature, labels, reward, trainable = False):
		
		if current_step<2:
			reduced_matrix = np.eye(self.feature_dim)
		else:
			reduced_matrix = self.calcu_reduced_matrix(remain_doc_feature)
		# print ("reduced_matrix shape : {}".format(self.reduced_matrix.get_shape()))
		# index = self.sess.run([self.d_index], feed_dict = {self.input_doc_feature : remain_doc_feature, self.reduced_matrix : reduced_matrix})

		# reward_standard = self.reward_standardized(reward)
		reward_standard = reward
		if trainable:
			loss, index = self.update_network(reward_standard, labels, remain_doc_feature, reduced_matrix)
		else:
			index = self.sess.run([self.d_index], feed_dict = {self.input_doc_feature : remain_doc_feature, self.reduced_matrix : reduced_matrix, self.dropout_keep_prob:1.0})
		return index


	def update_network(self, rewards, labels, doc_features, reduced_matrix):

		gradBuffer = self.sess.run(self.tvars)
		# print ("gradBuffer : {}".format(len(gradBuffer)))
		for ix, grad in enumerate(gradBuffer):
			gradBuffer[ix] = grad * 0
		# exit()
		# calcu grads
		tGrad, loss, index = self.sess.run([self.newGrads, self.loss, self.d_index], feed_dict={self.input_doc_feature : doc_features, self.input_y: np.array(labels), self.reduced_matrix: reduced_matrix,	
											self.tf_rt : np.array(rewards)[:,np.newaxis], self.dropout_keep_prob : 0.5})
		# print ("tGrad : {}".format(len(tGrad)))
		for ix, grad in enumerate(tGrad):
			gradBuffer[ix] += grad
		# print("gradBuffer[0] shape : {}".format(gradBuffer[0].shape))

		# print("gradBuffer[1] shape : {}".format(gradBuffer[1].shape))

		# print("gradBuffer[2] shape : {}".format(gradBuffer[2].shape))

		# print("gradBuffer[3] shape : {}".format(gradBuffer[3].shape))

		# print("gradBuffer[2] shape : {}".format(gradBuffer[4].shape))

		# print("gradBuffer[3] shape : {}".format(gradBuffer[5].shape))

		# print("gradBuffer[2] shape : {}".format(gradBuffer[6].shape))

		# print("gradBuffer[3] shape : {}".format(gradBuffer[7].shape))

		# print("gradBuffer[2] shape : {}".format(gradBuffer[8].shape))

		# print("gradBuffer[3] shape : {}".format(gradBuffer[9].shape))
		# print ("gradBuffer : {}".format(len(gradBuffer)))
		# exit()
		
		# update param
		_ = self.sess.run([self.train_op], feed_dict={self.W1Grad: gradBuffer[0], self.b1Grad: gradBuffer[1],self.W2Grad:gradBuffer[2], self.b2Grad:gradBuffer[3], 
							self.W3Grad: gradBuffer[4], self.b3Grad: gradBuffer[5],self.W4Grad:gradBuffer[6], self.b4Grad:gradBuffer[7],self.W5Grad:gradBuffer[8], self.b5Grad:gradBuffer[9]})
		
		# for ix, grad in enumerate(gradBuffer):
		# 	self.gradBuffer[ix] = grad * 0
		return loss, index


	def reset_network(self):		
		self.ep_docs, self.ep_label, self.ep_rs = [],[],[]

	def reward_standardized(self,rewards):
		standard_rewards = rewards
		standard_rewards -= np.mean(rewards)
		standard_rewards /= np.std(rewards)
		return standard_rewards

	def _discount_and_norm_rewards(self, ep_rs):
		discounted_ep_rs = np.zeros_like(ep_rs)
		runing_add = 0
		for t in reversed(range(0,len(ep_rs))):
			runing_add = runing_add*self.gamma+ep_rs[t]
			discounted_ep_rs[t] = runing_add

		discounted_ep_rs -= np.mean(discounted_ep_rs)
		discounted_ep_rs /= np.std(discounted_ep_rs)
		return discounted_ep_rs

	def save_param(self, write_str, timeDay):
		timeStamp = time.strftime("%Y%m%d%H%M%S", time.localtime(int(time.time())))
		folder = 'runs/' + timeDay+"/"+self.FLAGS.file_name
		# out_dir = folder +'/'+timeStamp+'_'+self.FLAGS.data+"_"+write_str

		out_dir = folder +'/'+self.FLAGS.data+"_"+write_str
		if not os.path.exists(folder):
			os.makedirs(folder)
		save_path = self.saver.save(self.sess, out_dir)
		print ("Model saved in file: ", save_path)


	def store_transition(self,doc_feature,label):
		# doc_feature mean currently selected document
		# label means currently selected document' label
		# reward means currently selected document' reward

		self.ep_docs.append(doc_feature)
		# self.ep_label.append(label[0])
		self.ep_label.append(label)
		# self.ep_rs.append(reward)