import torch
import torch.nn as nn

from utils.util import *

class Encoder(nn.Module):
	def __init__(self, seq_len=8):
		super(Encoder, self).__init__()

		self.seq_len = seq_len
		self.emb_size = 768
		
		self.position_emb = nn.Embedding(self.seq_len, self.emb_size // self.seq_len)
		self.transformer_layers = nn.ModuleList([TransformerLayer(self.emb_size // self.seq_len, 8, self.emb_size // self.seq_len * 4, 0.2) for _ in range(2)])

		self.dropout = nn.Dropout(p=0.1)
		self.apply(self._init_weights)

	def _init_weights(self, module):
		if isinstance(module, (nn.Linear, nn.Embedding)):
			module.weight.data.normal_(mean=0.0, std=0.02)
		elif isinstance(module, nn.LayerNorm):
			module.bias.data.zero_()
			module.weight.data.fill_(1.0)
		if isinstance(module, nn.Linear) and module.bias is not None:
			module.bias.data.zero_()

	def forward(self, batch_embeds):
		batch_embeds_seq = batch_embeds.view(batch_embeds.shape[0], self.seq_len, self.emb_size // self.seq_len)
		return batch_embeds_seq

	def forward_aux(self, batch_embeds):
		batch_embeds_seq = batch_embeds.view(batch_embeds.shape[0], batch_embeds.shape[1] * self.seq_len, self.emb_size // self.seq_len)
		return batch_embeds_seq