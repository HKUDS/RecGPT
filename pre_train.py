import argparse
import pickle
import os
import fsspec
import numpy as np

from accelerate import *

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from transformers import GPT2Config
from modeling_gpt2_rec import GPT2Model

from model_train import GPT4RecModel
from utils.data import *
from ae_trainer import *

import time

def main():

	dataloader_config = DataLoaderConfiguration(dispatch_batches=True)
	accelerator = Accelerator(dataloader_config=dataloader_config)
	device = accelerator.device

	parser = argparse.ArgumentParser()
	parser.add_argument("--lr", type=float, default=1e-4)
	parser.add_argument("--batch_size", type=int, default=12)
	parser.add_argument("--epoch", type=int, default=5)
	parser.add_argument("--tf_layer", type=int, default=3)
	args = parser.parse_args()

	args.n_embed = 768
	args.initializer_range = 0.02

	accelerator.print("-----Current Setting-----")
	accelerator.print(f"lr: {args.lr}")
	accelerator.print(f"batch size: {args.batch_size}")
	accelerator.print(f"gpt2 layer: {args.tf_layer}")

	with fsspec.open('./data/pre_train/train.pkl', "rb") as f:
		train_dict = pickle.load(f)

	accelerator.print("Records Number: " +  str(len(train_dict)))

	accelerator.print("-----Prepare Token----- ")

	with open('./data/pre_train/token_id_list.pkl', 'rb') as f:
		token_id_list = pickle.load(f)

	args.num_items = len(token_id_list)

	accelerator.print("-----Finish Token----- ")

	accelerator.print("Items Number: " +  str(args.num_items))

	gpt2 = GPT2Model(GPT2Config(n_layer=args.tf_layer))

	if accelerator.is_local_main_process:
		train_data = GPT2RecBatchTrainAuxData(train_dict, args.num_items)
		train_data.setTokenIdList(token_id_list)
	else:
		train_data = DummyDataset(len(train_dict))

	model = GPT4RecModel(args, gpt2)

	train_data_loader = DataLoader(train_data, args.batch_size, shuffle=True)

	model.to(device)

	optimizer = optim.AdamW(model.parameters(), lr=args.lr)

	model, optimizer, train_data_loader = accelerator.prepare(model, optimizer, train_data_loader)

	accelerator.print("Each epoch contains: " +  str(len(train_data_loader)))

	flag_10 = int(len(train_data_loader) * 0.1)
	flag_20 = int(len(train_data_loader) * 0.2)
	flag_30 = int(len(train_data_loader) * 0.3)
	flag_40 = int(len(train_data_loader) * 0.4)
	flag_50 = int(len(train_data_loader) * 0.5)
	flag_60 = int(len(train_data_loader) * 0.6)
	flag_70 = int(len(train_data_loader) * 0.7)
	flag_80 = int(len(train_data_loader) * 0.8)
	flag_90 = int(len(train_data_loader) * 0.9)
	flag_100 = len(train_data_loader)

	for epoch in range(args.epoch):
		model.train()
		train_rec_loss = 0.0

		counter = 0

		last_time = time.time()

		for batch_seq, batch_labels, batch_aux_embeds, aux_embed_mask in train_data_loader:
			counter += 1

			optimizer.zero_grad()

			batch_seq = batch_seq.to(device)
			batch_labels = batch_labels.to(device)
			batch_aux_embeds = batch_aux_embeds.to(device)
			aux_embed_mask = aux_embed_mask.to(device)

			accelerator.wait_for_everyone()

			loss_rec = model(batch_seq, batch_labels, batch_aux_embeds, aux_embed_mask)

			loss = loss_rec

			accelerator.backward(loss)

			optimizer.step()

			train_rec_loss += loss_rec.item()

			accelerator.wait_for_everyone()

			if counter == flag_10 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_01.pt')
				accelerator.print("Save ckpt for 0.1 data!")

			if counter == flag_20 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_02.pt')
				accelerator.print("Save ckpt for 0.2 data!")

			if counter == flag_30 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_03.pt')
				accelerator.print("Save ckpt for 0.3 data!")

			if counter == flag_40 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_04.pt')
				accelerator.print("Save ckpt for 0.4 data!")

			if counter == flag_50 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_05.pt')
				accelerator.print("Save ckpt for 0.5 data!")

			if counter == flag_60 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_06.pt')
				accelerator.print("Save ckpt for 0.6 data!")

			if counter == flag_70 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_07.pt')
				accelerator.print("Save ckpt for 0.7 data!")

			if counter == flag_80 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_08.pt')
				accelerator.print("Save ckpt for 0.8 data!")

			if counter == flag_90 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_09.pt')
				accelerator.print("Save ckpt for 0.9 data!")

			if counter == flag_100 and epoch == 0:
				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_10.pt')
				accelerator.print("Save ckpt for 1.0 data!")

			if counter % 100 == 1:
				accelerator.print(f"Current {counter} - Loss: {loss_rec.item():.4f}")

			if counter % 5000 == 0:
				accelerator.print(f"Current {counter} - Rec Loss: {(train_rec_loss / counter):.4f} - Time: {time.time() - last_time} s")
				last_time = time.time()

				torch.save(accelerator.unwrap_model(model).state_dict(), './ckpt/layer_' + str(args.tf_layer) + '_model_e' + str(epoch) + '.pt')
				
				accelerator.print("Save ckpt!")

		thread_train_rec_loss = torch.tensor([train_rec_loss / len(train_data_loader)]).to(device)
		gathered_train_rec_loss = accelerator.gather(thread_train_rec_loss)
		train_rec_loss = torch.mean(gathered_train_rec_loss)
		accelerator.print(f"Epoch {epoch} - Rec Loss: {train_rec_loss:.4f}")

if __name__ == "__main__":
	main()