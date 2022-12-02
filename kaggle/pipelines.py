import numpy as np
import pandas as pd

from sklego.preprocessing import ColumnSelector
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer, StandardScaler, QuantileTransformer, Normalizer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn_transformers import (
	FTLangdetectTransformer,
	PooledDeBertaTransformer
)

from config import MSFTDeBertaV3Config, FASTTEXT_MODEL_PATH
from english_utils import (
	number_of_unigrams,
	number_of_line_breaks,
	get_punctuation_error_fraction
)

def to_series(df: pd.DataFrame) -> pd.Series:
	assert df.shape[1] == 1
	return df.iloc[:, 0]


FEATURE_COLUMNS = ["full_text"]

feature_column_picker_pipe = Pipeline(
	steps=[
		("pick_full_text_column", ColumnSelector(FEATURE_COLUMNS)),
		("to_series", FunctionTransformer(to_series)),
	]
)

number_of_unigrams_pipe = Pipeline(
	steps=[
		("feature_column_picker", feature_column_picker_pipe),
		("count_unigrams", FunctionTransformer(number_of_unigrams)),
		("scale", StandardScaler())
	]
)

number_of_line_breaks_pipe = Pipeline(
	steps=[
		("feature_column_picker", feature_column_picker_pipe),
		("count_line_breaks", FunctionTransformer(number_of_line_breaks)),
		("scale", StandardScaler())
	]
)

english_score_pipe = Pipeline(
	steps=[
		("feature_column_picker", feature_column_picker_pipe),
		#                 ("english_scorer",  FunctionTransformer(ftlangdetect_english_score)),
		("english_scorer", FTLangdetectTransformer(model_path=FASTTEXT_MODEL_PATH)),
		("scale", StandardScaler())
	]
)

i_pipe = Pipeline(
	steps=[
		# pick the column
		("feature_column_picker", feature_column_picker_pipe),
		# count i vs I
		("i_I_counter", CountVectorizer(vocabulary=["i", "I"], lowercase=False, token_pattern=r"(?u)\b\w\b")),
		("union", FeatureUnion(
				[
					("l1normalizer", Normalizer(norm='l1')),
					("scaled_total_count", Pipeline(
							steps=[
								("sum_columns", FunctionTransformer(lambda x: np.sum(x, axis=1))),
								("std_scaler", StandardScaler())
							]
						)
					)
				]
			)
		)
	]
)

bad_punctuation_pipe = Pipeline(
	steps=[
		("feature_column_picker", feature_column_picker_pipe),
		("bad_punctuation_frac", FunctionTransformer(get_punctuation_error_fraction)),
	]
)

tf_idf_pipe = Pipeline(
	steps=[
		("feature_column_picker", feature_column_picker_pipe),
		("tf-idf", TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=0.01, max_df=0.99)),
	]
)

main_pipe = FeatureUnion(
	[
		("unigrams_count", number_of_unigrams_pipe),
		("line_breaks_count", number_of_line_breaks_pipe),
		("english_score", english_score_pipe),
		("i_vs_I", i_pipe),
		("bad_punctuation", bad_punctuation_pipe),
		("tf-idf", tf_idf_pipe)
	]
)

deberta_config = MSFTDeBertaV3Config(
		model_size="base",
		pooling="mean",
		inference_device="mps",
		batch_inference=False,
		output_device="cpu",
		inference_batch_size=4
	)


def make_deberta_pipeline(deberta_config):
	pooled_deberta_pipe = Pipeline(steps=[
			("feature_column_picker", feature_column_picker_pipe),
			("deberta_embedding", PooledDeBertaTransformer(deberta_config)),
		]
	)
	return pooled_deberta_pipe


if __name__ == "__main__":
	df = pd.DataFrame.from_dict({"full_text": [
				"Some text definitely in English.",
				"Another type of text, in English too."
			] * 4
		}
	)
	print(df.shape)
	# y_preds = main_pipe.fit_transform(df) #14
	# y_preds = pooled_deberta_pipe.fit_transform(df)  # 768
	y_preds = full_pipe.fit_transform(df)  # 782
	print(y_preds.shape)
	print("All Good!")