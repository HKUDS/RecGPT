import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np

class GPT4RecModel(nn.Module):
	def __init__(self, args, gpt2model):
		super(GPT4RecModel, self).__init__()

		self.num_items = args.num_items
		self.gpt2model = gpt2model
		self.n_embed = args.n_embed

		self.vocab_size = 15360 + 1
		self.padding_id = 15360

		self.loss_func = nn.CrossEntropyLoss()
		self.pred_head = nn.Linear(args.n_embed, self.vocab_size, bias=False)

		self.seq_len = 4

		self.linear_layer = nn.Linear(768 // self.seq_len, 768, bias=False)

		self.norm_seq = nn.LayerNorm(768)
		self.norm_aux = nn.LayerNorm(768)

	def setItemEmbed(self, iEmbed):
		self.iEmbed = iEmbed

	def setAutoEncoder(self, ae):
		self.ae = ae

	def forward_gpt(self, batch_seq, batch_aux_embeds, **kwargs):
		input_embeddings, attention_mask = self.embed(batch_seq, batch_aux_embeds)
		return self.gpt2model(inputs_embeds=input_embeddings, attention_mask=attention_mask, **kwargs)

	def forward(self, batch_seq, batch_labels, batch_aux_embeds, embed_mask, **kwargs):
		batch_aux_embeds = self.linear_layer(batch_aux_embeds)

		batch_aux_embeds = self.norm_aux(batch_aux_embeds)

		batch_aux_embeds = batch_aux_embeds * embed_mask
		transformer_outputs = self.forward_gpt(batch_seq, batch_aux_embeds, **kwargs)
		
		hidden_states = transformer_outputs[0]
		logits = self.pred_head(hidden_states)

		shift_logits = logits[..., :-4, :].contiguous()
		shift_labels = batch_labels[..., 4:].contiguous()

		loss_rec = self.loss_func(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

		return loss_rec

	def embed(self, input_ids, batch_aux_embeds):
		vocab_mask = (input_ids < self.vocab_size).long()
		
		vocab_ids = (input_ids * vocab_mask).clamp_(0, self.vocab_size-1)
		vocab_embeddings = self.gpt2model.wte(vocab_ids)
		
		vocab_embeddings = self.norm_seq(vocab_embeddings)

		vocab_embeddings = vocab_embeddings * vocab_mask.unsqueeze(-1)

		final_embeddings = vocab_embeddings + batch_aux_embeds

		attention_mask = (input_ids != self.vocab_size-1).long()

		return final_embeddings, attention_mask