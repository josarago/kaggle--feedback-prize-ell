import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression

# import lightgbm as lgb
import xgboost as xgb

from torch.utils.data import DataLoader
import torch.nn as nn

# project specific imports
from config import MSFTDeBertaV3Config
from torch_utils import Data
from pipelines import full_pipe


SCORE_COLUMNS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")


class ModelTrainer(ABC):
	def __init__(
			self,
			deberta_config,
			train_filename,
			test_filename,
			submission_filename,
			batch_inference=True,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS
	):
		self._deberta_config = deberta_config
		self._batch_inference = batch_inference
		self._train_filename = train_filename
		self._test_filename = test_filename
		self._submission_filename = submission_filename
		self._target_columns = target_columns
		self._feature_columns = feature_columns
		self._model = None
		self._pipeline = full_pipe

	def __repr__(self):
		repr_str = "'ModelTrainer' object"

	@classmethod
	def from_file(cls, train_filename, target_columns=SCORE_COLUMNS):
		if os.path.exists(train_filename):
			return cls(train_filename=train_filename)
		else:
			raise ValueError(f"file '{train_filename}' does not exist.")

	def load_data(self):
		df = pd.read_csv(self._train_filename)
		for column in self._target_columns:
			assert df[column].dtype == float
		df["partition"] = "train"
		return df

	def get_training_set(self, df):
		df_features = df[self._feature_columns]
		y = df[self._target_columns].values
		return df_features, y

	@staticmethod
	def split_data(self, df_features, y, test_size, random_state=42):
		(
			df_features_train,
			df_features_test,
			y_train,
			y_test
		) = train_test_split(
			df_features,
			y,
			test_size=test_size,
			random_state=random_state
		)
		return df_features_train, df_features_test, y_train, y_test

	def get_data_loader(X, y, batch_size, shuffle=True):
		# Instantiate training and test data
		data = Data(X, y)
		data_loader = DataLoader(dataset=data, batch_size=batch_size, shuffle=shuffle)
		return data_loader

	@abstractmethod
	def train(self, X, y):
		pass

	@staticmethod
	def recast_scores(y_pred):
		y_pred = np.round(y_pred * 2) / 2
		y_pred = np.min([y_pred, np.ones(y_pred.shape) * 5.0], axis=0)
		y_pred = np.max([y_pred, np.ones(y_pred.shape) * 1.0], axis=0)
		return y_pred

	@abstractmethod
	def predict(self, X, recast_scores=True):
		y_pred = self._model.predict(X)
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred

	@staticmethod
	def evaluate(y_true, y_pred):
		assert y_true.shape == y_pred.shape
		return np.mean([mean_squared_error(y_true[:, idx], y_pred[:, idx], squared=False) for idx in range(y_true.shape[1])])

	def evaluate_per_category(self, y_true, y_pred):
		eval_dict = dict()
		for idx, score_name in enumerate(self._target_columns):
			eval_dict[score_name] = mean_squared_error(y_true[:, idx], y_pred[:, idx], squared=False)
		return eval_dict

	def make_submission_df(self, recast_scores=True, write_file=True):
		print(f"loading test file from : '{self._test_filename}'")
		submission_df = pd.read_csv(self._test_filename)
		X_submission = self._pipeline.transform(submission_df)
		y_pred_submission = self.predict(X_submission, recast_scores=recast_scores)

		submission_df[self._target_columns] = y_pred_submission
		submission_df = submission_df[["text_id"] + self._target_columns]
		if write_file:
			print(f"Writing submission to: '{self._submission_filename}'")
			submission_df.to_csv(self._submission_filename, index=False)
		return submission_df


class SklearnRegressorTrainer(ModelTrainer):
	def __init__(
			self,
			deberta_config: MSFTDeBertaV3Config,
			model_type,
			fastext_model_path,
			batch_inference=True,
			train_filename=TRAIN_FILENAME,
			test_filename=TEST_FILENAME,
			submission_filename=SUBMISSION_FILENAME,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS,
	):

		assert model_type in ("dummy", "linear", "xgb"), "'model_type' must be either 'xgb', 'linear' or 'dummy'"
		self._deberta_config = deberta_config
		self._batch_inference = batch_inference
		self._model_type = model_type
		self._train_filename = train_filename
		self._test_filename = test_filename
		self._submission_filename = submission_filename
		self._target_columns = target_columns
		self._feature_columns = feature_columns
		self._model = None

	@classmethod
	def from_file(
			cls,
			deberta_config,
			train_filename,
			model_type,
			batch_inference,
			target_columns=SCORE_COLUMNS
	):
		if os.path.exists(train_filename):
			return cls(deberta_config, model_type, batch_inference=batch_inference, train_filename=train_filename)
		else:
			raise ValueError(f"file '{train_filename}' does not exist.")

	def train(self, X, y, params=None):
		# if self._model_type == "lgbm":
		# 	print("creating LightGBM regressor")
		# 	self._model = MultiOutputRegressor(lgb.LGBMRegressor(**params))
		if self._model_type == "xgb":
			print("creating XGBoost regressor")
			self._model = MultiOutputRegressor(xgb.XGBRegressor(**params if params else {}))
		elif self._model_type == "linear":
			print("creating linear model")
			self._model = LinearRegression()
		elif self._model_type == "dummy":
			print("creating dummy model")
			self._model = DummyRegressor(strategy="mean")
		self._model.fit(X, y)

	def predict(self, X, recast_scores=True):
		y_pred = self._model.predict(X)
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred


class NNTrainer(ModelTrainer):
	def __init__(
			self,
			deberta_config,
			device,
			fastext_model_path,
			batch_inference=True,
			train_filename=TRAIN_FILENAME,
			test_filename=TEST_FILENAME,
			submission_filename=SUBMISSION_FILENAME,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS,
	):
		self._deberta_config = deberta_config
		self._batch_inference = batch_inference
		self._device = device
		self._train_filename = train_filename
		self._test_filename = test_filename
		self._submission_filename = submission_filename
		self._target_columns = target_columns
		self._feature_columns = feature_columns
		self._model = None
		self._optimizer = None
		self._loss_fn = nn.MSELoss()

	@classmethod
	def from_file(cls, deberta_config, device, batch_inference, train_filename, target_columns=SCORE_COLUMNS):
		if os.path.exists(train_filename):
			return cls(deberta_config, device, batch_inference=batch_inference, train_filename=train_filename)
		else:
			raise ValueError(f"file '{train_filename}' does not exist.")

	def get_data_loader(self, X, y, bactch_size, shuffle=True):
		data = Data(X, y)
		data_loader = DataLoader(
			dataset=data,
			batch_size=bactch_size,
			shuffle=shuffle,
			collate_fn=lambda x: tuple(x_.to(self._device) for x_ in default_collate(x))
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
		self._model.to(self._device)
		self._optimizer = torch.optim.Adam(self._model.parameters(), lr=params["learning_rate"])
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

		min_val_loss = np.inf

		for epoch in range(params["num_epochs"]):
			_loss_values = dict(train=[])

			for X_train, y_train in train_data_loader:
				# Compute prediction error
				y_pred_train = self._model(X_train)
				train_loss = self._loss_fn(y_pred_train, y_train)
				_loss_values["train"].append(train_loss.item())

				# Backpropagation
				self._optimizer.zero_grad()
				train_loss.backward()
				self._optimizer.step()

			self._loss_values["train"].append(np.mean(_loss_values["train"]))
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
			self._loss_values["val"].append(np.mean(_loss_values["val"]))

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
		y_pred = self._model(Data.csr_to_torch(X).to(device)).cpu().detach().numpy()
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred

	def make_submission_df(self, recast_scores=True, write_file=True):
		"""
			implement batch prediction to avoid OOM errors?
		"""
		print(f"loading test file from : '{self._test_filename}'")
		submission_df = pd.read_csv(self._test_filename)
		X_submission = self._pipeline.transform(submission_df)
		y_pred_submission = self.predict(X_submission, recast_scores=recast_scores)

		submission_df[self._target_columns] = y_pred_submission
		submission_df = submission_df[["text_id"] + self._target_columns]
		if write_file:
			print(f"Writing submission to: '{self._submission_filename}'")
			submission_df.to_csv(self._submission_filename, index=False)
		return submission_df


if __name__ == "__main__":
	deberta_config = MSFTDeBertaV3Config(
		model_size="base",
		pooling="mean",
		inference_device="mps",
		output_device="mps",
		inference_batch_size=10
	)
	print(deberta_config)
	model_trainer = ModelTrainer(
		deberta_config,
		train_filename,
		test_filename,
		submission_filename,
		fastext_model_path,
		batch_inference=True,
		target_columns=SCORE_COLUMNS,
		feature_columns=FEATURE_COLUMNS
	)
