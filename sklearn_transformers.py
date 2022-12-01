import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

import fasttext

import torch
from torch_utils import MeanPooling
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoModel

from config import MSFTDeBertaV3Config
from english_utils import clean_special_characters

FASTTEXT_MODEL_PATH = "./input/fasttextmodel/lid.176.ftz"


def ft_langdetect(text: str, fasttext_model):
	labels, scores = fasttext_model.predict(text)
	label = labels[0].replace("__label__", '')
	score = min(float(scores[0]), 1.0)
	return {
		"lang": label,
		"score": score,
	}


def ftlangdetect_english_score(series: pd.Series, fasttext_model) -> np.array:
	""""""
	res = series.apply(lambda x: ft_langdetect(clean_special_characters(x), fasttext_model))
	return res.apply(lambda x: x['score'] if x["lang"] == 'en' else 1 - x['score']).values.reshape(-1, 1)


class FTLangdetectTransformer(BaseEstimator, TransformerMixin):
	def __init__(
			self,
			model_path=FASTTEXT_MODEL_PATH
	):
		"""
			This transformer outputs the predictions
			from a pretrained fasttext langdetect model
		"""
		self._model_path = model_path
		self._model = fasttext.load_model(self._model_path)

	def __repr__(self):
		return f"FTLangdetectTransformer from path '{self._model_path}'"

	def ft_langdetect(self, text: str):
		labels, scores = self._model.predict(text)
		label = labels[0].replace("__label__", '')
		score = min(float(scores[0]), 1.0)
		return {
			"lang": label,
			"score": score,
		}

	def fit(self, X, y=None):
		return self

	def transform(self, series: pd.Series) -> np.array:
		check_is_fitted(self, ['_model'])
		res = series.apply(lambda x: self.ft_langdetect(clean_special_characters(x)))
		return res.apply(lambda x: x['score'] if x["lang"] == 'en' else 1 - x['score']).values.reshape(-1, 1)


# class PooledDeBertaTransformer:
# 	def __init__(
# 			self,
# 			config: MSFTDeBertaV3Config,
# 			batch_inference=True,
#
# 	):
# 		"""
# 			This transformer outputs the last hidden layer
# 			after pooling, from a microsoft-deberta-v3-<size> model
# 		"""
# 		self.config = config
# 		self._batch_inference = batch_inference
# 		self.model = AutoModel.from_pretrained(
# 			self.config.model, config=self.config.config
# 		).to(
# 			self.config.inference_device
# 		)
# 		if self.config.gradient_checkpointing:
# 			self.model.gradient_checkpointing_enable()
# 		self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer)
# 		if config.pooling == 'mean':
# 			self.pool = MeanPooling()
#
# 	def fit(self, X, y=None):
# 		return self
#
# 	def prepare_input(self, input_texts):
# 		"""
# 			this will truncate the longest essays.
# 			Consider adding some sliding window logic and pooling/aggregation to capture the full text?
# 		"""
# 		if isinstance(input_texts, pd.Series):
# 			texts_list = input_texts.values.tolist()
# 		else:
# 			texts_list = input_texts
# 		inputs = self.tokenizer.batch_encode_plus(
# 			texts_list,
# 			return_tensors=None,
# 			add_special_tokens=True,
# 			max_length=self.config.tokenizer_max_length,
# 			pad_to_max_length=True,
# 			truncation=True
# 		)
# 		for k, v in inputs.items():
# 			inputs[k] = torch.tensor(v, dtype=torch.long)
# 		return inputs
#
# 	@torch.no_grad()
# 	def feature(self, inputs):
# 		self.model.eval()
# 		#         last_hidden_states = self.model(**{k: v.to(self.config._inference_device) for k, v, in inputs.items()})[0]
# 		last_hidden_states = self.model(
# 			**{k: v.to(self.config.inference_device) for k, v, in inputs.items()}
# 		).last_hidden_state
# 		feature = self.pool(
# 			last_hidden_states,
# 			inputs['attention_mask'].to(self.config.inference_device)
# 		)
# 		return feature
#
# 	def batch_transform(self, series):
# 		preds = []
# 		data_loader = DataLoader(
# 			dataset=series,
# 			batch_size=self.config._inference_batch_size,
# 			shuffle=False,
# 			collate_fn=lambda x: self.prepare_input(x).to(self.config.inference_device),
# 			num_workers=1
# 		)
# 		#         bar = tqdm(enumerate(data_loader), total=len(data_loader))
# 		#         for step, inputs in bar:
# 		for inputs in data_loader:
# 			with torch.no_grad():
# 				y_preds = self.feature(inputs)
#
# 			preds.append(y_preds.to(self.config.output_device))
# 		predictions = np.concatenate(preds)
# 		return predictions
#
# 	def simple_transform(self, series):
# 		inputs = self.prepare_input(series).to(self.config.inference_device)
# 		y_preds = self.feature(inputs).to(self.config.output_device)
# 		return y_preds
#
# 	def transform(self, series):
# 		check_is_fitted(self, ['model', 'tokenizer'])
# 		if self._batch_inference:
# 			return self.batch_transform(series)
# 		else:
# 			return self.simple_transform(series)

def prepare_input(
		config,
		tokenizer,
		input_texts
	):
	"""
		this will truncate the longest essays.
		Consider adding some sliding window logic and pooling/aggregation to capture the full text?
	"""
	if isinstance(input_texts, pd.Series):
		texts_list = input_texts.values.tolist()
	else:
		texts_list = input_texts
	inputs = tokenizer.batch_encode_plus(
		texts_list,
		return_tensors=None,
		add_special_tokens=True,
		max_length=config.tokenizer_max_length,
		pad_to_max_length=True,
		truncation=True
	)
	for k, v in inputs.items():
		inputs[k] = torch.tensor(v, dtype=torch.long)
	return inputs


@torch.no_grad()
def feature(model, config, inputs):
	model.eval()
	#         last_hidden_states = self.model(**{k: v.to(self.config._inference_device) for k, v, in inputs.items()})[0]
	last_hidden_states = model(
		**{k: v.to(config.inference_device) for k, v, in inputs.items()}
	).last_hidden_state
	feature = MeanPooling()(
		last_hidden_states,
		inputs['attention_mask'].to(config.inference_device)
	)
	return feature

# class PooledDeBertaTransformer:

n_samples = 800
batch_size = 8

if __name__ == "__main__":
	ftl_transformer = FTLangdetectTransformer(model_path=FASTTEXT_MODEL_PATH)
	print(ftl_transformer.__class__)

	print(ftl_transformer.fit_transform(
		pd.Series([
			"Some text definitely in English",
			"Un texte absolument pas en Anglais"
		])
	))
	DEVICE = "cpu"
	deberta_config = MSFTDeBertaV3Config(
		model_size="base",
		pooling="mean",
		inference_device=DEVICE,
		output_device="cpu",
		inference_batch_size=batch_size
	)

	# pooled_deberta_transformer = PooledDeBertaTransformer(
	# 	deberta_config,
	# 	batch_inference=True
	# )
	#
	# print(pooled_deberta_transformer.simple_transform(
	# 	pd.Series([
	# 		"Some text definitely in English",
	# 		"Another type of text, in English too"
	# 	])
	# ).shape)



	# print(pooled_deberta_transformer.batch_transform(
	# 	pd.Series([
	# 		"Some text definitely in English",
	# 		"Another type of text, in English too"
	# 	])
	# ).shape)
	tokenizer = AutoTokenizer.from_pretrained(deberta_config.tokenizer)
	model = AutoModel.from_pretrained(deberta_config.model).to(
			deberta_config.inference_device
		)
	series = pd.Series(
		[
			"Some text definitely in English.",
			"Another type of text, in English too."
		] * (n_samples // 2)
	)
	print(series.shape)
	preds = []
	data_loader = DataLoader(
		dataset=series,
		batch_size=deberta_config._inference_batch_size,
		shuffle=False,
		# num_workers=4
	)
	#         bar = tqdm(enumerate(data_loader), total=len(data_loader))
	#         for step, inputs in bar:
	for _series in tqdm(data_loader):
		_inputs = prepare_input(
			deberta_config,
			tokenizer,
			_series
		).to(deberta_config.inference_device)
		with torch.no_grad():
			y_preds = feature(model, deberta_config, _inputs)

		preds.append(y_preds.to(deberta_config.output_device))
	predictions = np.concatenate(preds)
	print(predictions.shape)





