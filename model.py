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

	def set_fsq_encoder(self, encoder):
		self.encoder = encoder

	def set_fsq_decoder(self, decoder):
		self.decoder = decoder

	def setItemEmbed(self, iEmbed):
		self.iEmbed = iEmbed

	def setAutoEncoder(self, ae):
		self.ae = ae

	def forward_gpt(self, batch_seq, batch_aux_embeds, **kwargs):
		input_embeddings, attention_mask = self.embed(batch_seq, batch_aux_embeds)
		return self.gpt2model(inputs_embeds=input_embeddings, attention_mask=attention_mask, **kwargs)

	def forward(self, batch_seq, batch_labels, batch_aux, ae, **kwargs):
		batch_aux_embeds, embed_mask = ae.encode_aux(batch_aux, self.num_items, batch_seq.device)
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

	def setTrie(self, trie):
		self.trie = trie

	def setMapDict(self, mapDict):
		self.mapDict = mapDict

	def predict_aux(self, batch_seq, batch_lengths, batch_aux):
		max_length = torch.max(batch_lengths)
		batch_seq = batch_seq[..., -max_length:]
		batch_aux = batch_aux[..., -(max_length // 4):]

		batch_aux_embeds, embed_mask = self.ae.encode_aux(batch_aux, self.num_items, batch_seq.device)
		batch_aux_embeds = self.linear_layer(batch_aux_embeds)

		batch_aux_embeds = self.norm_aux(batch_aux_embeds)

		batch_aux_embeds = batch_aux_embeds * embed_mask
		input_embeddings, attn_mask = self.embed(batch_seq, batch_aux_embeds)
		output = self.gpt2model(inputs_embeds=input_embeddings, attention_mask=attn_mask)
		logits = self.pred_head(output[0])
		logits = logits[..., -4:,:]

		beams = [(None, torch.zeros(batch_seq.shape[0], dtype=torch.long, device=batch_seq.device))]

		for i in range(4):
			new_beams = []
			temp_beams = []
			for generated, score in beams:
				current_logits = logits[..., i, :]
		
				current_logits = torch.log_softmax(current_logits / 0.1, dim=-1)
				if generated == None:
					beam_width = 10
					for j in range(batch_seq.shape[0]):
						candidates = self.trie.get_possible_characters_after_prefix([])
						mask = torch.ones_like(current_logits[j]).bool()
						mask[candidates] = False
						current_logits[j] = torch.masked_fill(current_logits[j], mask, -1e7)
				else:
					batch_prefix = generated[:, -i:]
					beam_width = 5
					for j in range(batch_seq.shape[0]):
						prefix = batch_prefix[j].tolist()
						candidates = self.trie.get_possible_characters_after_prefix(prefix)
						mask = torch.ones_like(current_logits[j]).bool()
						mask[candidates] = False
						current_logits[j] = torch.masked_fill(current_logits[j], mask, -1e7)

				top_log_probs, top_tokens = torch.topk(current_logits, beam_width, dim=-1)

				for j in range(beam_width):
					if generated == None:
						new_generated = top_tokens[:, j].unsqueeze(-1)
					else:
						new_generated = torch.cat((generated, top_tokens[:, j].unsqueeze(-1)), dim=-1)
					new_score = score + top_log_probs[:, j]
					temp_beams.append((new_generated, new_score))

			beam_list4sort = []
			for j in range(batch_seq.shape[0]):
				in_batch_beam = []
				for idx in range(len(temp_beams)):
					in_batch_beam.append((temp_beams[idx][0][j], temp_beams[idx][1][j]))
				in_batch_beam = sorted(in_batch_beam, key=lambda x: x[1], reverse=True)
				beam_list4sort.append(in_batch_beam)

			for j in range(len(temp_beams)):
				temp_generated = []
				temp_score = []
				for idx in range(batch_seq.shape[0]):
					temp_generated.append(beam_list4sort[idx][j][0])
					temp_score.append(beam_list4sort[idx][j][1])
				temp_generated = torch.stack(temp_generated)
				temp_score = torch.stack(temp_score)
				new_beams.append((temp_generated, temp_score))

			for index_ in range(len(new_beams)):
				_, score = new_beams[index_]

			beams = new_beams[:20]

		pred_logits = torch.full((batch_seq.shape[0], self.num_items), -1e8)

		counter = 0

		for i in range(len(beams)):
			generated, score = beams[i]
			for j in range(batch_seq.shape[0]):
				temp_list = generated[j].tolist()
				try:
					pred_logits[j][self.mapDict[temp_list[0]][temp_list[1]][temp_list[2]][temp_list[3]]] = score[j]
					counter += 1
				except KeyError:
					continue

		return pred_logits

