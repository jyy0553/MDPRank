from __future__ import division
import pandas as pd 
import subprocess
import platform,os
import sklearn
import numpy as np




# def ranking_process(RL_L2R, data):
def calcu_MAP_function(label_list):
	R = 0
	ap=0
	is_one = 1
	index = 1
	for i in label_list:
		if i>0:
			R+=1

	if R == 0:
		return 0

	for i in label_list:
		if i > 0:
			ap += (1.0*is_one)/(1.0*index)
			is_one += 1
		index += 1
	return ap/(1.0*R)


def calcu_NDCG_function(label_list):
	# print (label_list)
	# AtP=20
	rank_list = []
	for item in label_list:
		rank_list.append(item[0])
	DCG = []
	IDCG = []
	correct_candidates = 0

	for i in rank_list:
		if i>0:
			correct_candidates+=1
	if correct_candidates == 0:
		return 0

	# calcu all DCG
	for index in range(1,len(label_list)+1):
		current_DCG = (np.power(2.0,rank_list[index-1])-1.0)/np.log2(index+1)
		DCG.append(current_DCG)

	# calcu all IDCG
	idel_label_list = np.array(rank_list)
	idel_label_list = -np.sort(-idel_label_list)

	for index in range(1,len(idel_label_list)+1):
		current_IDCG = (np.power(2.0, idel_label_list[index-1])-1.0)/np.log2(index+1)
		IDCG.append(current_IDCG)

	# calcu NDCG@1
	NDCG_at_1 = float(sum(DCG[:1])/sum(IDCG[:1]))

	# calcu NDCG@3
	NDCG_at_3 = float(sum(DCG[:3])/sum(IDCG[:3]))

	# calcu NDCG@5
	NDCG_at_5 = float(sum(DCG[:5])/sum(IDCG[:5]))

	# calcu NDCG@10
	NDCG_at_10 = float(sum(DCG[:10])/sum(IDCG[:10]))
	
	# calcu NDCG@20
	NDCG_at_20 = float(sum(DCG[:20])/sum(IDCG[:20]))
	# print (DCG)
	# print (IDCG)
	return NDCG_at_1, NDCG_at_3, NDCG_at_5,NDCG_at_10,NDCG_at_20

def calcu_ERR_function(label_list):
	AtP = 20
	if len(label_list) < AtP:
		AtP = len(label_list)

	correct_candidates = 0
	for i in label_list:
		if i>0:
			correct_candidates+=1
	if correct_candidates == 0:
		return 0

	gmax = max(label_list)
	ERR = 0
	for r in range(1, AtP+1):
		pp_r = 1
		for i in range(1,r):
			R_i = float((np.power(2.0,label_list[i-1])-1.0)/np.power(2.0,gmax))
			pp_r *= (1.0 - R_i)
		R_r = float((np.power(2.0,label_list[r-1])-1.0)/np.power(2.0,gmax))
		pp_r *= R_r
		ERR += (1.0/r)*pp_r

	return ERR


def calcu_P_function(label_list):
	AtP = 20
	# print (label_list)

	if len(label_list) < AtP:
		AtP = len(label_list)

	true_num = 0
	false_num = 0
	index = 1
	for i in label_list:
		# print (i)
		if index > AtP:
			break
		if i > 0:
			true_num += 1
		else:
			false_num += 1
		index += 1
	# print ("true_num : {}".format(true_num))
	p = float((1.0*true_num)/(1.0*AtP))
	return p

def evaluation_ranklists(label_collection):
	# print (label_collection)
	calcu_MAP = []
	calcu_NDCG_at_1 = []
	calcu_NDCG_at_3 = []
	calcu_NDCG_at_5 = []
	calcu_NDCG_at_10 = []	
	calcu_NDCG_at_20 = []
	calcu_MRR = []
	calcu_P = []
	# print (label_collection)
	# exit()
	j = 0
	for qid_labels in label_collection:
		# print ("J : {}".format(j))
		# j+=1
		# print (qid_labels)
		calcu_MAP.append(calcu_MAP_function(qid_labels))
		NDCG_at_1,NDCG_at_3,NDCG_at_5,NDCG_at_10,NDCG_at_20 = calcu_NDCG_function(qid_labels)
		calcu_NDCG_at_1.append(NDCG_at_1)
		calcu_NDCG_at_3.append(NDCG_at_3)
		calcu_NDCG_at_5.append(NDCG_at_5)
		calcu_NDCG_at_10.append(NDCG_at_10)
		calcu_NDCG_at_20.append(NDCG_at_20)
		calcu_MRR.append(calcu_ERR_function(qid_labels))
		calcu_P.append(calcu_P_function(qid_labels))
	# print ("calcu_NDCG_at_1 collection : {}".format(calcu_NDCG_at_1))
	# exit()
	MAP = np.mean(calcu_MAP)
	NDCG_at_1 = np.mean(calcu_NDCG_at_1)
	NDCG_at_3 = np.mean(calcu_NDCG_at_3)
	NDCG_at_5 = np.mean(calcu_NDCG_at_5)
	NDCG_at_10 = np.mean(calcu_NDCG_at_10)
	NDCG_at_20 = np.mean(calcu_NDCG_at_20)
	MRR = np.mean(calcu_MRR)
	P = np.mean(calcu_P)
	return MAP,NDCG_at_1,NDCG_at_3,NDCG_at_5,NDCG_at_10,NDCG_at_20,MRR,P