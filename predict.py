import argparse
import pickle
import os
import fsspec
import numpy as np
import random
import time

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import GPT2Config
from modeling_gpt2_rec import GPT2Model

from model import GPT4RecModel
from utils.data import *
from ae_trainer import *
from utils.Trie import *

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--dataset", type=str, default='yelp')
	parser.add_argument("--test_batch_size", type=int, default=96)
	parser.add_argument("--tf_layer", type=int, default=3)
	parser.add_argument("--gpu", type=str, default='0')
	parser.add_argument("--cold", type=bool, default=False)
	parser.add_argument("--ckpt_path", type=str, default='./ckpt/recgpt_layer_3_weight.pt')
	args = parser.parse_args()

	device = torch.device("cuda:" + args.gpu)

	args.n_embed = 768
	args.initializer_range = 0.02

	data_path = os.path.join('./data/', args.dataset)

	if args.cold == True:
		with open(data_path + '/cold_test.pkl', "rb") as f:
			test_dict = pickle.load(f)	
	else: 
		with open(data_path + '/test.pkl', "rb") as f:
			test_dict = pickle.load(f)

	print("Records Number: " +  str(len(test_dict)))

	with open(data_path + '/item_text_embeddings.npy', "rb") as f:
		item_text_embeddings = np.load(f)
		item_text_embeddings = torch.tensor(item_text_embeddings).float().to(device)

	args.num_items = item_text_embeddings.shape[0]

	vae_data = VAEData(args.num_items)
	vae_data_loader = DataLoader(vae_data, 2048, shuffle=False)

	ae = AutoEncoder(seq_len=4)
	ae.load_state_dict(torch.load('./vae_ckpt/vae_len4_fsq88865_ep90.pt', map_location=device))
	ae.setItemEmbeddings(item_text_embeddings)
	ae.to(device)

	ae.eval()
	token_id_list = []
	with torch.no_grad():
		for batch_ids in vae_data_loader:
			_, quant_indices = ae.encode(batch_ids)
			token_id_list.extend(quant_indices.cpu().tolist()) 

	trie = Trie()
	for codes in token_id_list:
		trie.insert(codes)

	gpt2 = GPT2Model(GPT2Config(n_layer=args.tf_layer))

	test_data = GPT2RecBatchTestAuxData(test_dict, args.num_items)
	test_data.setTokenIdList(token_id_list)

	model = GPT4RecModel(args, gpt2)
	
	model.load_state_dict(torch.load(args.ckpt_path, map_location=device))
	print("load successfully!")

	model.setItemEmbed(item_text_embeddings)
	model.setAutoEncoder(ae)
	model.setTrie(trie)

	code_to_item_dict = {}
	for i in range(len(token_id_list)):
		code_1 = token_id_list[i][0]
		code_2 = token_id_list[i][1]
		code_3 = token_id_list[i][2]
		code_4 = token_id_list[i][3]
		if code_1 not in code_to_item_dict.keys():
			code_to_item_dict[code_1] = {}
		if code_2 not in code_to_item_dict[code_1].keys():
			code_to_item_dict[code_1][code_2] = {}
		if code_3 not in code_to_item_dict[code_1][code_2].keys():
			code_to_item_dict[code_1][code_2][code_3] = {}
		code_to_item_dict[code_1][code_2][code_3][code_4] = i

	model.setMapDict(code_to_item_dict)

	test_data_loader = DataLoader(test_data, args.test_batch_size, shuffle=False)

	model.to(device)
	model.eval()
	cur_hit_1, cur_ndcg_1, cur_hit_3, cur_ndcg_3, cur_hit_5, cur_ndcg_5 = eval_in_epoch(test_data_loader, model, args, test_data, device)

	print('hit@1 = %.4f, ndcg@1 = %.4f, hit@3 = %.4f, ndcg@3 = %.4f, hit@5 = %.4f, ndcg@5 = %.4f' % (cur_hit_1, cur_ndcg_1, cur_hit_3, cur_ndcg_3, cur_hit_5, cur_ndcg_5))

def eval_in_epoch(test_data_loader, model, args, test_data, device):
	with torch.no_grad():
		epHr_1, epNdcg_1 = [0] * 2
		epHr_3, epNdcg_3 = [0] * 2
		epHr_5, epNdcg_5 = [0] * 2
		batch_ratings_1 = []
		batch_ratings_3 = []
		batch_ratings_5 = []
		ground_truths = []

		for batch_idx, batch_seq, target_iids, batch_lengths, batch_aux in test_data_loader:
			
			batch_seq = batch_seq.to(device)
			batch_lengths = batch_lengths.to(device)
			batch_aux = batch_aux.to(device)

			batch_pred = model.predict_aux(batch_seq, batch_lengths, batch_aux)

			_, batch_rate_1 = torch.topk(batch_pred, 1)
			_, batch_rate_3 = torch.topk(batch_pred, 3)
			_, batch_rate_5 = torch.topk(batch_pred, 5)

			batch_ratings_1.append(batch_rate_1.cpu())
			batch_ratings_3.append(batch_rate_3.cpu())
			batch_ratings_5.append(batch_rate_5.cpu())
			ground_truth = []
			target_iids = target_iids.numpy().tolist()
			for ti in target_iids:
				ground_truth.append([ti])
			ground_truths.append(ground_truth)


		data_pair_1 = zip(batch_ratings_1, ground_truths)
		data_pair_3 = zip(batch_ratings_3, ground_truths)
		data_pair_5 = zip(batch_ratings_5, ground_truths)
		
		eval_results_1 = []
		eval_results_3 = []
		eval_results_5 = []
		
		for _data in data_pair_1:
			hr, ndcg = eval_batch(_data, 1)
			eval_results_1.append((hr, ndcg))
		for _data in data_pair_3:
			hr, ndcg = eval_batch(_data, 3)
			eval_results_3.append((hr, ndcg))
		for _data in data_pair_5:
			hr, ndcg = eval_batch(_data, 5)
			eval_results_5.append((hr, ndcg))
		
		for batch_result in eval_results_1:
			epHr_1 += batch_result[0]
			epNdcg_1 += batch_result[1]
		for batch_result in eval_results_3:
			epHr_3 += batch_result[0]
			epNdcg_3 += batch_result[1]
		for batch_result in eval_results_5:
			epHr_5 += batch_result[0]
			epNdcg_5 += batch_result[1]

		return epHr_1/len(test_data), epNdcg_1/len(test_data), epHr_3/len(test_data), epNdcg_3/len(test_data), epHr_5/len(test_data), epNdcg_5/len(test_data)

def eval_batch(data, topk):
	sorted_items = data[0].numpy()
	ground_true = data[1]
	r = get_label(ground_true, sorted_items)

	result_hr = hr(r, topk)
	result_ndcg = ndcg(ground_true, r, topk)

	return result_hr, result_ndcg

def get_label(test_data, pred_data):
	r = []
	for i in range(len(test_data)):
		ground_true = test_data[i]
		predict_topk = pred_data[i]
		pred = list(map(lambda x: x in ground_true, predict_topk))
		pred = np.array(pred).astype("float")
		r.append(pred)
	return np.array(r).astype('float')

def hr(r, k):
	right_pred = r[:, :k].sum(1)
	right_pred = np.clip(right_pred, a_min=0, a_max=1)
	hr = np.sum(right_pred)
	return hr

def ndcg(test_data, r, k):
	assert len(r) == len(test_data)
	pred_data = r[:, :k]

	test_matrix = np.zeros((len(pred_data), k))
	for i, items in enumerate(test_data):
		length = k if k <= len(items) else len(items)
		test_matrix[i, :length] = 1
	max_r = test_matrix
	idcg = np.sum(max_r * 1. / np.log2(np.arange(2, k + 2)), axis=1)
	dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
	dcg = np.sum(dcg, axis=1)
	idcg[idcg == 0.] = 1.
	ndcg = dcg / idcg
	ndcg[np.isnan(ndcg)] = 0.
	return np.sum(ndcg)

if __name__ == "__main__":
	main()