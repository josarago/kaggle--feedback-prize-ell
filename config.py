import os
import torch

MODELS_DIR_PATH = './input/fb3models'


class MSFTDeBertaV3Config:
	def __init__(
			self,
			model_size,
			pooling,
			inference_device,
			output_device="cpu",
			inference_batch_size=100,

	):
		"""
			used manage the deberta model configuration
		"""
		assert model_size in ["base", "large", "xlarge", "xxlarge"]
		self._model_size = model_size
		self._models_dir_path = MODELS_DIR_PATH
		self._path_prefix = "microsoft-deberta-v3-"
		self._base_dir_path = os.path.join(self._models_dir_path, self._path_prefix + self._model_size)
		self.gradient_checkpointing = False
		self._inference_batch_size = inference_batch_size
		self.inference_device = inference_device if inference_device else torch.device(
			"cuda:0" if torch.cuda.is_available() else "cpu")
		self.output_device = output_device
		self.tokenizer_max_length = 512
		self._seed = 42
		self.pooling = pooling
		assert self.pooling == "mean", "we removed all other implementations other than 'mean'."

	@property
	def config(self):
		return os.path.join(self._base_dir_path, "config")

	@property
	def model(self):
		return os.path.join(self._base_dir_path, "model")

	@property
	def tokenizer(self):
		return os.path.join(self._base_dir_path, "tokenizer")

	def __repr__(self):
		print(f"MSFTDeBertaV3Config config object with: model_size: '{self._model_size}' and inference device: '{self.inference_device}'")

