#coding=utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
# from pre_dateset import batch_gen_with_point_wise,load,prepare,batch_gen_with_single
from pre_process import load_data, get_batch, get_each_query_length, get_batch_with_test
from model_new import QRL_L2R

from evaluation import evaluation_ranklists
import random
# import evaluation as evaluation_test
# import evaluation_test
# import cPickle as pickle
import pickle
from sklearn.model_selection import train_test_split
import configure
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


from functools import wraps

def log_time_delta(func):
    @wraps(func)
    def _deco(*args, **kwargs):
        start = time.time()
        ret = func(*args, **kwargs)
        end = time.time()
        delta = end - start
        print( "%s runed %.2f seconds"% (func.__name__,delta))
        return ret
    return _deco

FLAGS = configure.flags.FLAGS
FLAGS.flag_values_dict()
# FLAGS._parse_flags()

now = int(time.time()) 
timeArray = time.localtime(now)
timeStamp = time.strftime("%Y%m%d%H%M%S", timeArray)
timeDay = time.strftime("%Y%m%d", timeArray)

log_dir = 'log/'+ timeDay
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
data_file = log_dir + '/test_' + FLAGS.data + timeStamp
para_file = log_dir + '/test_' + FLAGS.data + timeStamp + '_para'
precision = data_file + 'precise'

def predict(RL_L2R, dataset):

	reward_sum = 0
	label_collection = []
	for data in get_batch_with_test(dataset, FLAGS.feature_dim):
		doc_feature = data[0]
		doc_label = data[1]
		doc_len = data[2]
		for step in range(doc_len):
			immediate_rewards = calcu_immediate_reward(step, doc_label)
			selected_doc_index = RL_L2R.choose_doc(step, doc_feature, doc_label, immediate_rewards)
				
			# selected_doc_index = RL_L2R.choose_doc(step,doc_feature)
			current_doc = doc_feature[selected_doc_index]
			current_label = doc_label[selected_doc_index]
			doc_feature, doc_label = get_candidata_feature_label(selected_doc_index, doc_feature, doc_label) 
			# reward = calcu_reward(step, current_label)
			# RL_L2R.store_transition(current_doc,current_label,reward)			
			RL_L2R.store_transition(current_doc,current_label)
		reward = calcu_reward(RL_L2R.ep_label)
		label_collection.append(RL_L2R.ep_label)
		RL_L2R.reset_network()
		reward_sum += reward

	return label_collection, reward_sum

def get_candidata_feature_label(index, doc_feature, doc_label):
	r_doc_feature = np.delete(doc_feature, index, 0)
	r_doc_label = np.delete(doc_label, index, 0)
	return r_doc_feature, r_doc_label


def calcu_reward(ep_rs):
	# discounted_ep_rs = np.zeros_like(ep_rs)
	running_add = 0
	DCG = calcu_DCG(ep_rs) 
	for t in range(len(ep_rs)):
		running_add += (FLAGS.reward_decay**t) * DCG[t]
		# discounted_ep_rs[t] = runing_add

	return running_add

def calcu_DCG(ep_labels):
	DCG = []
	for i in range(len(ep_labels)):
		if i == 0:
			DCG.append(np.power(2.0, ep_labels[i]) - 1.0)
		else:
			DCG.append((np.power(2.0, ep_labels[i]) - 1.0)/np.log2(i+1))
	# print ("DCG : {}".format(DCG))
	return DCG

def calcu_immediate_reward(current_step, labels):
	decay_rate = FLAGS.reward_decay
	DCG = []
	# print (decay_rate**current_step)
	if current_step == 0:
		for i in range(len(labels)):
			temp_DCG = np.power(2.0, labels[i]) - 1.0
			DCG.append((decay_rate**current_step)*temp_DCG)
	else:
		for i in range(len(labels)):
			temp_DCG = (np.power(2.0, labels[i]) - 1.0)/np.log2(current_step+1)
			DCG.append((decay_rate**current_step)*temp_DCG)

	return DCG

	
def train():
	file_name = FLAGS.file_name
	train_set = load_data("../pre_prosess/OHSUMED/"+file_name+"/trainingset.txt")
	test_set = load_data("../pre_prosess/OHSUMED/"+file_name+"/testset.txt")
	valid_set = load_data("../pre_prosess/OHSUMED/"+file_name+"/validationset.txt")
	each_query_length = get_each_query_length()

	log = open(precision,"w")
	log.write(str(FLAGS.__flags)+'\n')

	RL_L2R = QRL_L2R(
			feature_dim = FLAGS.feature_dim,
			learn_rate = FLAGS.learning_rate,
			reward_decay = FLAGS.reward_decay,
			FLAGS = FLAGS
			)
			
	max_ndcg_1 = 0.020
	max_ndcg_10 = 0.02
	max_reward = 1
	# loss_max = 0.3
	for i in range(FLAGS.num_epochs):
		print ("\nepoch "+str(i)+"\n")
		j = 1
		# reward_sum = 0
		# training process
		for data in get_batch(train_set, FLAGS.feature_dim):
			doc_feature = data[0]
			doc_label = data[1]
			doc_len = data[2]
			qid = data[3]
			# print ("doc_label : {}".format(doc_label))
			for step in range(doc_len):
				immediate_rewards = calcu_immediate_reward(step, doc_label)
				selected_doc_index = RL_L2R.choose_doc(step, doc_feature, doc_label, immediate_rewards, True)
				current_doc = doc_feature[selected_doc_index]
				current_label = doc_label[selected_doc_index]
				doc_feature, doc_label = get_candidata_feature_label(selected_doc_index, doc_feature, doc_label) 
				# print (current_label)
				RL_L2R.store_transition(current_doc,current_label)

			# print ("RL_L2R.ep_label : {}".format(RL_L2R.ep_label))
			reward = calcu_reward(RL_L2R.ep_label)
			# print (reward)
			# idel_reward, idel_features = calcu_idel_reward(RL_L2R.ep_docs, RL_L2R.ep_label)
			# ep_rs_norm, loss = RL_L2R.learn(reward)		
			# ep_rs_norm, loss = RL_L2R.learn(reward, idel_reward, idel_features)
			# loss = RL_L2R.learn(reward)
			RL_L2R.reset_network()
			# reward_sum += reward
			print ("training, qid :{} with_length : {}, reward : {}".format(qid, doc_len, reward))
			# break

		# train evaluation
		train_predict_label_collection, train_reward = predict(RL_L2R, train_set)	
		train_MAP, train_NDCG_at_1, train_NDCG_at_3, train_NDCG_at_5, train_NDCG_at_10, train_NDCG_at_20, train_MRR, train_P = evaluation_ranklists(train_predict_label_collection)
		train_result_line = "## epoch {}, train MAP : {}, train_NDCG_at_1 : {}, train_NDCG_at_3 : {}, train_NDCG_at_5 : {}, train_NDCG_at_10 : {}, train_NDCG_at_20 : {}, train_MRR@20 : {}, train_P@20 : {}, \ntrain_reward : {}".format(i, train_MAP, train_NDCG_at_1, train_NDCG_at_3, train_NDCG_at_5, train_NDCG_at_10, train_NDCG_at_20, train_MRR, train_P, train_reward[0])

		print (train_result_line)		
		log.write(train_result_line+"\n")	


		
		# valid evaluation
		valid_predict_label_collection, valid_reward = predict(RL_L2R, valid_set)		
		valid_MAP, valid_NDCG_at_1, valid_NDCG_at_3, valid_NDCG_at_5, valid_NDCG_at_10, valid_NDCG_at_20, valid_MRR, valid_P = evaluation_ranklists(valid_predict_label_collection)
		valid_result_line = "## epoch {}, valid_MAP : {}, valid_NDCG_at_1 : {}, valid_NDCG_at_3 : {}, valid_NDCG_at_5 : {}, valid_NDCG_at_10 : {}, valid_NDCG_at_20 : {}, valid_MRR@20 : {}, valid_P@20 : {}, \nvalid_reward : {}".format(i, valid_MAP, valid_NDCG_at_1, valid_NDCG_at_3, valid_NDCG_at_5, valid_NDCG_at_10, valid_NDCG_at_20, valid_MRR, valid_P, valid_reward[0])
		print (valid_result_line)
		log.write(valid_result_line+"\n")

		# save param		
		if valid_reward > max_reward:
			max_reward = valid_reward[0] 
			write_str = str(max_reward) +"_"+str(valid_NDCG_at_1)+"_"+str(valid_NDCG_at_10)			
			RL_L2R.save_param(write_str, timeDay)


		# if valid_NDCG_at_1 > max_ndcg_1 and valid_NDCG_at_10 > max_ndcg_10:
		# 	max_ndcg_1 = valid_NDCG_at_1
		# 	max_ndcg_10 = valid_NDCG_at_10
		# 	write_str = str(max_ndcg_1)+"_"+str(max_ndcg_10)
		# 	RL_L2R.save_param(write_str, timeDay)

		# test evaluation
		test_predict_label_collection, test_reward = predict(RL_L2R, test_set)				

		test_MAP, test_NDCG_at_1, test_NDCG_at_3, test_NDCG_at_5, test_NDCG_at_10, test_NDCG_at_20, test_MRR, test_P = evaluation_ranklists(test_predict_label_collection)
		test_result_line = "## test_MAP : {}, test_NDCG_at_1 : {}, test_NDCG_at_3 : {}, test_NDCG_at_5 : {}, test_NDCG_at_10 : {}, test_NDCG_at_20 : {}, test_MRR@20 : {}, test_P@20 : {}, \ntest_reward : {}".format(test_MAP, test_NDCG_at_1, test_NDCG_at_3, test_NDCG_at_5, test_NDCG_at_10, test_NDCG_at_20, test_MRR, test_P, test_reward[0])
		print (test_result_line)
		log.write(test_result_line+"\n\n")


	# test process
	
	test_predict_label_collection, test_reward = predict(RL_L2R, test_set)				

	test_MAP, test_NDCG_at_1, test_NDCG_at_3, test_NDCG_at_5, test_NDCG_at_10, test_NDCG_at_20, test_MRR, test_P = evaluation_ranklists(test_predict_label_collection)
	test_result_line = "## test_MAP : {}, test_NDCG_at_1 : {}, test_NDCG_at_3 : {}, test_NDCG_at_5 : {}, test_NDCG_at_10 : {}, test_NDCG_at_20 : {}, test_MRR@20 : {}, test_P@20 : {}, \ntest_reward : {}".format(test_MAP, test_NDCG_at_1, test_NDCG_at_3, test_NDCG_at_5, test_NDCG_at_10, test_NDCG_at_20, test_MRR, test_P, test_reward[0])
	print (test_result_line)
	log.write(test_result_line+"\n")

	log.write("\n")
	log.flush()
	log.close()
	
	# label_collection = []
	# for data in get_batch_with_test(test_set):
	# 	doc_feature = data[0]
	# 	doc_label = data[1]
	# 	doc_len = data[2]
	# 	for step in range(doc_len):
	# 		selected_doc_index = RL_L2R.choose_doc(step,doc_feature)
	# 		current_doc = doc_feature[selected_doc_index]
	# 		current_label = doc_label[selected_doc_index]
	# 		doc_feature, doc_label = get_candidata_feature_label(selected_doc_index, doc_feature, doc_label) 
	# 		reward = calcu_reward(step, current_label)
	# 		RL_L2R.store_transition(current_doc,current_label,reward)
	# 	label_collection.append(RL_L2R.ep_lable)
			

			# if j == 5: 
			# 	exit()
			# j+=1
	# line1 = " {}:epoch: map_train{}".format(i,map_NDCG0_NDCG1_ERR_p_train)
	# log.write(line1+"\n")
	# line = " {}:epoch: map_test{}".format(i,map_NDCG0_NDCG1_ERR_p_test)
	# log.write(line+"\n")
	# log.write("\n")
	# log.flush()
	# log.close()
	# exit()


if __name__ == "__main__":
	train()
			