import torch
import torch.nn as nn

from utils.util import *

class Decoder(nn.Module):
	def __init__(self, seq_len=8):
		super(Decoder, self).__init__()

		self.seq_len = seq_len
		self.emb_size = 768
		
		self.position_emb = nn.Embedding(self.seq_len, self.emb_size // self.seq_len)
		self.transformer_layers = nn.ModuleList([TransformerLayer(self.emb_size // self.seq_len, 8, self.emb_size // self.seq_len * 4, 0.2) for _ in range(2)])
		self.linear_layer = nn.Sequential(nn.Linear(self.emb_size // self.seq_len, self.emb_size), nn.ReLU(inplace=True), nn.Linear(self.emb_size, self.emb_size))

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
		batch_embeds_seq = batch_embeds + self.position_emb.weight.unsqueeze(0).repeat(batch_embeds.shape[0], 1, 1)
		x = self.dropout(batch_embeds_seq)
		for transformer in self.transformer_layers:
			x = transformer(x, None)
		output = x[:, -1, :]
		output = self.linear_layer(output)
		return output