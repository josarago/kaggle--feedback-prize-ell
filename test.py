import pandas as pd
from config import MSFTDeBertaV3Config
from sklearn_transformers import PooledDeBertaTransformer

if __name__ == "__main__":
	deberta_config = MSFTDeBertaV3Config(
		model_size="base",
		pooling="mean",
		inference_device="mps",
		output_device="mps",
		inference_batch_size=10
	)

	pooled_deberta_transformer = PooledDeBertaTransformer(
		deberta_config
	)

	print(pooled_deberta_transformer.fit_transform(
		pd.Series([
			"Some text definitely in English",
			"Another type of text, in English too"
		])
	))