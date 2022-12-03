import os
import torch
import platform

KAGGLE_ROOT_DIR = "."
INPUT_DIR = "input"
CHALLENGE_NAME = "feedback-prize-english-language-learning"
SUBMISSION_DIR = "working"

FASTTEXT_MODEL_PATH = os.path.join(KAGGLE_ROOT_DIR, INPUT_DIR,  "fasttextmodel/lid.176.ftz")
MODELS_DIR_PATH = os.path.join(KAGGLE_ROOT_DIR, INPUT_DIR,  "fb3models")


def get_default_inference_device():
	if "arm" in platform.platform():
		return "mps:0"
	else:
		if torch.cuda.is_available():
			return "cuda:0"
	return "cpu"


class MSFTDeBertaV3Config:
	def __init__(
			self,
			model_size,
			pooling,
			inference_device=None,
			batch_inference=True,
			output_device="cpu",
			inference_batch_size=100,

	):
		"""
			used manage the deberta model configuration
		"""
		assert model_size in ["base", "large", "xlarge", "xxlarge"]
		self._model_size = model_size
		assert pooling == "mean", "we removed all other implementations other than 'mean'."
		self.pooling = pooling
		self.inference_device = inference_device if inference_device else get_default_inference_device()
		self._batch_inference = batch_inference
		self._inference_batch_size = inference_batch_size
		self._models_dir_path = MODELS_DIR_PATH
		self._path_prefix = "microsoft-deberta-v3-"
		self._base_dir_path = os.path.join(self._models_dir_path, self._path_prefix + self._model_size)
		self.gradient_checkpointing = False
		self.output_device = output_device
		self.tokenizer_max_length = 512

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
		return f"MSFTDeBertaV3Config config object with: model_size: '{self._model_size}' and inference device: '{self.inference_device}'"


DEFAULT_DEBERTA_CONFIG = MSFTDeBertaV3Config(
	"base",
	"mean",
	inference_device=None,
	batch_inference=True,
	output_device="cpu",
	inference_batch_size=8,
)

