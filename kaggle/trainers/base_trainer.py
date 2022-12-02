import os
from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import FeatureUnion

# import lightgbm as lgb
import xgboost as xgb

from torch.utils.data import DataLoader
import torch.nn as nn

# project specific imports
from config import (
	MSFTDeBertaV3Config,
	DEFAULT_DEBERTA_CONFIG,
	KAGGLE_ROOT_DIR,
	INPUT_DIR,
	CHALLENGE_NAME,
	SUBMISSION_DIR
)
from torch_utils import Data
from sklearn_transformers import PooledDeBertaTransformer
from pipelines import full_pipe


SCORE_COLUMNS = ("cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions")
FEATURE_COLUMNS = ("full_text",)


class ModelTrainer(ABC):
	def __init__(
			self,
			deberta_config: MSFTDeBertaV3Config = DEFAULT_DEBERTA_CONFIG,
			batch_inference=True,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS,
			train_file_name=None,
			test_file_name=None,
			submission_filename=None,
	):
		self._deberta_config = deberta_config
		self._batch_inference = batch_inference
		self._challenge_name = CHALLENGE_NAME
		self._train_filename = train_file_name if train_file_name else os.path.join(KAGGLE_ROOT_DIR, INPUT_DIR, CHALLENGE_NAME, "train.csv")
		self._test_filename = test_file_name if test_file_name else os.path.join(KAGGLE_ROOT_DIR, INPUT_DIR, CHALLENGE_NAME, "test.csv")
		self._submission_filename = submission_filename if submission_filename else os.path.join(KAGGLE_ROOT_DIR, SUBMISSION_DIR, "submission.csv")
		self._target_columns = list(target_columns)
		self._feature_columns = list(feature_columns)
		self._model = None
		self._pipeline = full_pipe

	def __repr__(self):
		return "'ModelTrainer' object"

	# @classmethod unused
	# def from_file(cls, train_filename=None):
	# 	return cls(DEFAULT_DEBERTA_CONFIG, train_filename=train_filename)

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
	def split_data(df_features, y, test_size, random_state=42):
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


if __name__ == "__main__":
	print(os.getcwd())
	deberta_config = MSFTDeBertaV3Config(
		model_size="base",
		pooling="mean",
		inference_device="mps",
		batch_inference=True,
		output_device="cpu",
		inference_batch_size=10
	)
	print(deberta_config)
	model_trainer = ModelTrainer(
		deberta_config,
		target_columns=SCORE_COLUMNS,
		feature_columns=FEATURE_COLUMNS
	)
	df = model_trainer.load_data()
	df_features, y = model_trainer.get_training_set(df.iloc[:20])
	df_features_train, df_features_test, y_train, y_test = model_trainer.split_data(df_features, y, test_size=0.33)
	X_train = model_trainer._pipeline.fit_transform(df_features_train)
	print(X_train.shape)
	print("Excuse you.")
