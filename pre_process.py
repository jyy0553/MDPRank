#coding=utf-8
import sys
import os
import json
import math
import copy

import pandas as pd
import collections
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
# from nltk.stem.Snowball import SnowballStemmer 
# from gensim.models.keyedvectors import KeyedVectors
from random import *
import numpy as np
import importlib
import pickle
import sklearn
from sklearn import feature_extraction  
from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer 
import heapq
from sklearn.model_selection import train_test_split
from functools import wraps
import time
# stoplist = stopwords.words('english')
#creat a list which contains query and its candidate document set
#each candidate document set have many document which
# wlem = WordNetLemmatizer()



def preprocess_dataset(file_name, outfile_name):
	'''
	function: Process the original data set
	out data format: (qid, docid, feature(25 dims), label)

	file_name: the path of original dataset
	outfile_name: the path of the processed file

	'''
	f = open(file_name, "r")
	f_out = open(outfile_name,'w')
	for line in f.readlines():
		
		lineList = line.strip().split(" ")
		
		feature = ""
		for i in range(2,27):
			feature += lineList[i].split(":")[1]+ " "
			
		feature = feature.strip()
		qid = lineList[1].split(":")[1]
		label = lineList[0]
		docid = qid+"_"+lineList[len(lineList)-1]
		
		write_str = qid + "\t" + docid + "\t" + feature + "\t" +label + "\n"
		f_out.write(write_str)
		
	f.close()
	f_out.close()

# for i in range(3,6):
# 	preprocess_dataset("../dataset/OHSUMED/Fold"+str(i)+"/testset.txt","../pre_prosess/OHSUMED/Fold"+str(i)+"/testset.txt")

# 	preprocess_dataset("../dataset/OHSUMED/Fold"+str(i)+"/trainingset.txt","../pre_prosess/OHSUMED/Fold"+str(i)+"/trainingset.txt")

# 	preprocess_dataset("../dataset/OHSUMED/Fold"+str(i)+"/validationset.txt","../pre_prosess/OHSUMED/Fold"+str(i)+"/validationset.txt")


def get_max_single_query_len(df):

	'''
	function: get Maximum length of candidate documents for all pueries 
	df: all data
	return: a scalar
	'''
	dic = {}
	qid_len = []
	for index,row in df.iterrows():
		qid = row["query_ID"]
		docid = row["doc_ID"]
		feature = row["feature"]
		label = row["flag"]
		# print (label)
		# exit()
		if qid in dic:
			# dic[qid].append((qid,docid,feature,label))			
			dic[qid].append((feature,label))
		else:
			dic.update({qid:[]})
			dic[qid].append((feature,label))

	for qid in dic:		
		pairs = dic[qid]
		qid_batchlen = len(pairs)
		qid_len.append(qid_batchlen)
	# print (qid_len)
	max_single_query_len = max(qid_len)
	return max_single_query_len

def load_data(data_path):
	'''
	function: load data from the processed file	
	data_path: the path of data
	'''

	data_file = os.path.join(data_path)
	data = pd.read_csv(data_file,header = None,sep="\t",names=["query_ID","doc_ID","feature","flag"],quoting =3).fillna('')
	
	return data



# train = load_data("pre_prosess/OHSUMED/All/OHSUMED.txt")
# max_single_query_len = get_max_single_query_len(train)
# print (max_single_query_len)
# print (max_single_query_len)
# exit()
# def get_query_length():

def get_each_query_length():

	df = load_data("../pre_prosess/OHSUMED/All/OHSUMED.txt")
	dic = {}
	qid_len = []
	for index,row in df.iterrows():
		qid = row["query_ID"]
		docid = row["doc_ID"]
		feature = row["feature"]
		label = row["flag"]
		# print (label)
		# exit()
		if qid in dic:
			# dic[qid].append((qid,docid,feature,label))			
			dic[qid].append((feature,label))
		else:
			dic.update({qid:[]})
			dic[qid].append((feature,label))

	for qid in dic:		
		pairs = dic[qid]
		qid_batchlen = len(pairs)

		# print ("qid :{} with length :{}".format(qid,qid_batchlen))
		qid_len.append(qid_batchlen)
	
	return np.array(qid_len)


def get_each_query_length_for_model():

	df = load_data("../pre_prosess/OHSUMED/All/OHSUMED.txt")
	dic = {}
	qid_len = []
	for index,row in df.iterrows():
		qid = row["query_ID"]
		docid = row["doc_ID"]
		feature = row["feature"]
		label = row["flag"]
		# print (label)
		# exit()
		if qid in dic:
			# dic[qid].append((qid,docid,feature,label))			
			dic[qid].append((feature,label))
		else:
			dic.update({qid:[]})
			dic[qid].append((feature,label))

	for qid in dic:		
		pairs = dic[qid]
		qid_batchlen = len(pairs)

		# print ("qid :{} with length :{}".format(qid,qid_batchlen))
		qid_len.append(qid_batchlen)
	# print (qid_len)
	# max_single_query_len = max(qid_len)
	train = []
	valid = []
	test = []
	for i in range(len(qid_len)):
		if i <= 63:
			train.append(qid_len[i])
		elif i<=84 and i>63:
			valid.append(qid_len[i])
		else:
			test.append(qid_len[i])
	# print (train)
	# print (len(train))
	# print (valid)
	# print (len(valid))
	# print (test)
	# print (len(test))
	return_list = []
	return_list.append(train)
	return_list.append(valid)
	return_list.append(test)
	# print (return_list)
	# print (return_list[0][1])
	return np.array(return_list)


# train = load_data("pre_prosess/OHSUMED/All/OHSUMED.txt")
# single_query_len = get_each_query_length()
# print (single_query_len[0])
# print (single_query_len[0][3])
# print ()
# print (single_query_len.shape)


def encode_feature(feature_str):
	'''
	function: converts feature string into list
	feature_str: a feature string
	return: a feature list
	'''
	feature_list = []
	items = feature_str.strip().split(" ")
	for i in items:
		feature_list.append(float(i))
	# print (feature_list)
	return feature_list

def encode_qid_batchlen(num):
	return np.zeros([num,2], dtype = float)

# def encode_label(labels):
# 	r_list = []
# 	print (labels)
	# for i in range(labels):
	# 	print(i)
	
def feature_normal(dic,feature_dim):
	mean = np.zeros(feature_dim)
	doc_num = 0

	# calcu mean
	for qid in dic:
		pairs = dic[qid]
		for pair in pairs:
			mean += pair[0]
			doc_num += 1
	mean /= doc_num

	# calcu standard deviation
	sigma=np.zeros(feature_dim)
	for qid in dic:
		pairs = dic[qid]
		for pair in pairs:
			sigma += (pair[0]-mean) * (pair[0]-mean)
	sigma /= doc_num
	sigma = np.sqrt(sigma)
	for i in range(len(sigma)):
		if sigma[i] == 0:
			sigma[i] = 1
	
	# calcu normal data
	for qid in dic:
		pairs = dic[qid]
		for pair in pairs:
			pair[0] = (pair[0]-mean)/sigma
		
	return dic

def get_batch(df,feature_dim):
	'''
	function: get all candidate document of the query
	df: all data
	batch_size: Maximum length of candidate documents for all pueries
	'''
	input_num =2
	dic = {}
	for index,row in df.iterrows():
		qid = row["query_ID"]
		docid = row["doc_ID"]
		feature = encode_feature(row["feature"])
		label = int(row["flag"])
		# exit()
		if qid in dic:
			dic[qid].append([feature,label])
		else:
			dic.update({qid:[]})
			dic[qid].append([feature,label])

	dic = feature_normal(dic, feature_dim)

	# exit()
	for qid in dic:
		if qid == 8:
			continue
		out_list = []
		pairs = dic[qid]
		shuffle(pairs)
		qid_batchlen = len(pairs)
		yield [np.array([pair[j] for pair in pairs]) for j in range(input_num)] + [qid_batchlen] + [np.array(qid)]
		

# train = load_data("pre_prosess/OHSUMED/All/OHSUMED.txt")

# for i in get_batch(train,25):
# 	print ("*************")
# 	break


def get_batch_with_test(df,feature_dim):
	'''
	function: get all candidate document of the query
	df: all data
	batch_size: Maximum length of candidate documents for all pueries
	'''
	input_num =2
	dic = {}
	for index,row in df.iterrows():
		qid = row["query_ID"]
		docid = row["doc_ID"]
		feature = encode_feature(row["feature"])
		label = int(row["flag"])
		# exit()
		if qid in dic:
			dic[qid].append([feature,label])
		else:
			dic.update({qid:[]})
			dic[qid].append([feature,label])

	dic = feature_normal(dic, feature_dim)
			
	for qid in dic:
		if qid == 8:
			continue
		out_list = []
		pairs = dic[qid]
		qid_batchlen = len(pairs)
		yield [np.array([pair[j] for pair in pairs]) for j in range(input_num)] + [qid_batchlen] + [np.array(qid)]
		