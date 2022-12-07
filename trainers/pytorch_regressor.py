import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.dataloader import default_collate

from trainers.base_trainer import (
	SCORE_COLUMNS,
	FEATURE_COLUMNS,
	ModelTrainer
)
from config import MSFTDeBertaV3Config
from torch_utils import Data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split # replace


class EllActivation(nn.Module):
	def __init__(self, force_half_points=False):
		super().__init__()
		self._force_half_points = force_half_points

	def forward(self, x):
		y = torch.sigmoid(x) * 4. + 1.
		if self._force_half_points:
			y = torch.round(y * 2) / 2
		return y


class SequentialNeuralNetwork(nn.Module):
	def __init__(self, X, y, hidden_dims=None, n_hidden=3, force_half_points=False):
		super(SequentialNeuralNetwork, self).__init__()
		# parameters
		self._input_dim = X.shape[1]
		self._output_dim = y.shape[1]
		assert hidden_dims is not None or n_hidden is not None, "either n_hidden or hidden_dims should be non null"
		if hidden_dims:
			print("hidden_dims:", hidden_dims)
			self._hidden_dims = hidden_dims
			self._n_hidden = len(self._hidden_dims)

		if n_hidden is not None:
			print("n_hidden:", n_hidden)
			self._n_hidden = n_hidden
			self._alpha = (np.log(self._output_dim) / np.log(self._input_dim)) ** (1 / (self._n_hidden + 1))
			self._hidden_dims = [
				int(np.round(self._input_dim ** (self._alpha ** i))) for i in np.arange(1, self._n_hidden + 1)]

		self._force_half_points = force_half_points
		self._model = nn.Sequential()
		if self._n_hidden > 0:
			for dim_in, dim_out in zip([self._input_dim] + self._hidden_dims, self._hidden_dims):
				linear_layer = nn.Linear(dim_in, dim_out, bias=True)
				#                 nn.init.xavier_uniform(linear_layer.weight)
				self._model.append(linear_layer)

				self._model.append(nn.GELU())
			self._model.append(nn.Linear(self._hidden_dims[-1], self._output_dim, bias=True))
			self._model.append(EllActivation(force_half_points=self._force_half_points))
		else:
			self._model.append(nn.Linear(self._input_dim, self._output_dim, bias=True))
			self._model.append(EllActivation(force_half_points=self._force_half_points))
		print(self._model)

	def forward(self, x):
		return self._model(x)


class NNTrainer(ModelTrainer):
	def __init__(
			self,
			fastext_model_path,
			deberta_config: MSFTDeBertaV3Config,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS,
			train_file_name=None,
			test_file_name=None,
			submission_filename=None,
	):
		super().__init__(
			fastext_model_path,
			deberta_config,
			target_columns=target_columns,
			feature_columns=feature_columns,
			train_file_name=train_file_name,
			test_file_name=test_file_name,
			submission_filename=submission_filename,
		)
		# pytorch specific
		self._optimizer = None
		self._loss_fn = nn.MSELoss()
		self._loss_values = dict()
		self._training_device = self._deberta_config.inference_device

	def get_data_loader(self, X, y, bactch_size, shuffle=True):
		data = Data(X, y)
		data_loader = DataLoader(
			dataset=data,
			batch_size=bactch_size,
			shuffle=shuffle,
			collate_fn=lambda x: tuple(x_.to(self._training_device) for x_ in default_collate(x))
		)
		return data_loader

	def train(self, X, y, params):
		self._model = SequentialNeuralNetwork(
			X,
			y,
			hidden_dims=params["hidden_dims"],
			n_hidden=params["n_hidden"],
			force_half_points=params["force_half_points"]
		)
		self._model.to(self._training_device)
		self._optimizer = torch.optim.Adam(
			self._model.parameters(),
			lr=params["learning_rate"]
		)
		self._loss_values = dict()
		self._loss_values["train"] = []
		if params["with_validation"]:
			print("Using validation")
			self._loss_values["val"] = []
			X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params["val_size"])
			train_data_loader = self.get_data_loader(X_train, y_train, params["batch_size"], params["shuffle"])
			val_data_loader = self.get_data_loader(X_val, y_val, params["batch_size"], params["shuffle"])
		else:
			train_data_loader = self.get_data_loader(X, y, params["batch_size"], params["shuffle"])

		for epoch in range(params["num_epochs"]):
			_loss_values = dict(train=[])

			for X_train, y_train in train_data_loader:
				# Compute prediction error
				y_pred_train = self._model(X_train.to(self._training_device))
				train_loss = self._loss_fn(y_pred_train, y_train.to(self._training_device))
				_loss_values["train"].append(train_loss.item())

				# Backpropagation
				self._optimizer.zero_grad()
				train_loss.backward()
				self._optimizer.step()

			self._loss_values["train"].append(np.sqrt(np.mean(_loss_values["train"])))
			if params["with_validation"]:
				_loss_values["val"] = []
				self._model.eval()  # Optional when not using Model Specific layer
				for X_val, y_val in val_data_loader:
					# Forward Pass
					y_pred_val = self._model(X_val)
					# Find the Loss
					val_loss = self._loss_fn(y_pred_val, y_val)
					# Calculate Loss
					_loss_values["val"].append(val_loss.item())
			self._loss_values["val"].append(np.sqrt(np.mean(_loss_values["val"])))

		print("Training Complete")

	def plot_loss_values(self):
		fig, ax = plt.subplots(figsize=(20, 5))
		ax.plot(np.array(self._loss_values["train"]), label="training")
		if "val" in self._loss_values.keys():
			ax.plot(np.array(self._loss_values["val"]), label="validation")
		ax.set_xlabel("Epochs")
		ax.set_ylabel("Losses")
		ax.legend()
		plt.show()

	def predict(self, X, recast_scores=True):
		y_pred = self._model(Data.csr_to_torch(X).to(self._training_device)).cpu().detach().numpy()
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred
