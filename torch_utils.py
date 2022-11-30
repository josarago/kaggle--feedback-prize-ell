import numpy as np
from scipy.sparse import csr_matrix

import torch
import torch.nn as nn
from torch.utils.data import Dataset, random_split


class Data(Dataset):
	@classmethod
	def csr_to_torch(cls, X):
		if isinstance(X, csr_matrix):
			_X = X.todense()
		else:
			_X = X
		Xt = torch.from_numpy(_X.astype(np.float32))
		return Xt

	def __init__(self, X, y):
		self.X = self.csr_to_torch(X)
		self.y = torch.from_numpy(y.astype(np.float32))
		self.len = self.X.shape[0]

	def __getitem__(self, index):
		return self.X[index], self.y[index]

	def __len__(self):
		return self.len


class MeanPooling(nn.Module):
	def __init__(self, clamp_min=1e-9):
		super(MeanPooling, self).__init__()
		self._clamp_min = clamp_min

	def forward(self, last_hidden_state, attention_mask):
		input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
		sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, 1)
		sum_mask = input_mask_expanded.sum(1)
		sum_mask = torch.clamp(sum_mask, min=self._clamp_min)
		mean_embeddings = sum_embeddings / sum_mask
		return mean_embeddings
