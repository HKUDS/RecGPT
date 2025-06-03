from torch import nn
import torch.nn.functional as F
import torch
import math

class TransformerLayer(nn.Module):
	def __init__(self, hidden_size, num_heads, feed_forward_size, dropout_rate):
		super().__init__()
		self.attention = MultiHeadAttention(num_heads=num_heads, hidden_size=hidden_size, dropout=dropout_rate)
		self.feed_forward = PositionwiseFeedForward(hidden_size=hidden_size, d_ff=feed_forward_size, dropout=dropout_rate)
		self.input_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
		self.output_sublayer = ResidualConnection(hidden_size=hidden_size, dropout=dropout_rate)
		self.dropout = nn.Dropout(p=dropout_rate)

	def forward(self, x, mask):
		x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
		x = self.output_sublayer(x, self.feed_forward)
		return self.dropout(x)

class TransformerEmbedding(nn.Module):
	def __init__(self, item_num, emb_size, max_len, dropout=0.1):
		"""
		:param vocab_size: total vocab size
		:param embed_size: embedding size of token embedding
		:param dropout: dropout rate
		"""
		super().__init__()
		self.token_emb = nn.Embedding(item_num, emb_size, padding_idx=0)
		self.position_emb = nn.Embedding(max_len, emb_size)
		self.dropout = nn.Dropout(p=dropout)
		self.emb_size = emb_size

	def forward(self, batch_seqs):
		batch_size = batch_seqs.size(0)
		pos_emb = self.position_emb.weight.unsqueeze(
			0).repeat(batch_size, 1, 1)
		x = self.token_emb(batch_seqs) + pos_emb

		return self.dropout(x)
	
class MultiHeadAttention(nn.Module):
	def __init__(self, num_heads, hidden_size, dropout=0.1):
		super().__init__()
		assert hidden_size % num_heads == 0

		self.d_k = hidden_size // num_heads
		self.n_h = num_heads

		self.linear_layers = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(3)])
		self.output_linear = nn.Linear(hidden_size, hidden_size)

		self.dropout = nn.Dropout(p=dropout)
	
	def _cal_attention(self, query, key, value, mask=None, dropout=None):
		scores = torch.matmul(query, key.transpose(-2, -1)) \
				 / math.sqrt(query.size(-1))

		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e9)

		p_attn = F.softmax(scores, dim=-1)

		if dropout is not None:
			p_attn = dropout(p_attn)

		return torch.matmul(p_attn, value), p_attn

	def forward(self, query, key, value, mask=None):
		batch_size = query.size(0)

		# 1) Do all the linear projections in batch from d_model => h x d_k
		query, key, value = [l(x).view(batch_size, -1, self.n_h, self.d_k).transpose(1, 2)
							 for l, x in zip(self.linear_layers, (query, key, value))]

		# 2) Apply attention on all the projected vectors in batch.
		x, attn = self._cal_attention(query, key, value, mask=mask, dropout=self.dropout)

		# 3) "Concat" using a view and apply a final linear.
		x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_h * self.d_k)

		return self.output_linear(x)
		
class PositionwiseFeedForward(nn.Module):
	def __init__(self, hidden_size, d_ff, dropout=0.1):
		super(PositionwiseFeedForward, self).__init__()
		self.w_1 = nn.Linear(hidden_size, d_ff)
		self.w_2 = nn.Linear(d_ff, hidden_size)
		self.dropout = nn.Dropout(dropout)
		self.activation = nn.GELU()

	def forward(self, x):
		return self.w_2(self.dropout(self.activation(self.w_1(x))))

class ResidualConnection(nn.Module):
	def __init__(self, hidden_size, dropout):
		super(ResidualConnection, self).__init__()
		self.norm = nn.LayerNorm(hidden_size)
		self.dropout = nn.Dropout(dropout)

	def forward(self, x, sublayer):
		"Apply residual connection to any sublayer with the same size."
		return x + self.dropout(sublayer(self.norm(x)))
	
    