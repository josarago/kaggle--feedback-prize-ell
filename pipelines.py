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

from config import DEFAULT_DEBERTA_CONFIG, FASTTEXT_MODEL_PATH
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


def make_english_score_pipe(model_path=FASTTEXT_MODEL_PATH):
	english_score_pipe = Pipeline(
		steps=[
			("feature_column_picker", feature_column_picker_pipe),
			("english_scorer", FTLangdetectTransformer(model_path=model_path)),
			("scale", StandardScaler())
		]
	)
	return english_score_pipe


def make_deberta_pipe(deberta_config=DEFAULT_DEBERTA_CONFIG):
	pooled_deberta_pipe = Pipeline(steps=[
			# this is upsetting as hell but somehow the only way I can make this pipeline to work
			("index_resetter", FunctionTransformer(lambda _df: _df.reset_index())),
			("feature_column_picker", feature_column_picker_pipe),
			("deberta_embedding", PooledDeBertaTransformer(deberta_config)),
		]
	)
	return pooled_deberta_pipe


def make_features_pipeline(
	fastext_model_path=FASTTEXT_MODEL_PATH,
	deberta_config=DEFAULT_DEBERTA_CONFIG
):
	features_pipeline = FeatureUnion(
		[
			("unigrams_count", number_of_unigrams_pipe),
			("line_breaks_count", number_of_line_breaks_pipe),
			("english_score", make_english_score_pipe(fastext_model_path)),
			("i_vs_I", i_pipe),
			("bad_punctuation", bad_punctuation_pipe),
			("tf-idf", tf_idf_pipe),
			("deberta_pipe", make_deberta_pipe(deberta_config))

		]
	)
	return features_pipeline


if __name__ == "__main__":
	df = pd.DataFrame.from_dict({"full_text": [
				"Some text definitely in English.",
				"Another type of text, in English too."
			] * 4
		}
	)
	print(df.shape)
	features_pipeline = make_features_pipeline()
	y_preds = features_pipeline.fit_transform(df)
	print(y_preds.shape)
	print("All Good!")