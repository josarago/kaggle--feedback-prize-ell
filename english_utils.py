import re
import numpy as np
import pandas as pd

PUNCTUATION_CHARACTERS = "'!(),-./:;?" + '"'


def count_punctuation_characters(text, characters=PUNCTUATION_CHARACTERS):
	"""
	count punctuation characters
	"""
	return np.sum([text.count(char) for char in characters])


def count_missing_trailing_whitespaces(text):
	"""
	find missing trailing spaces after `.,!?` except if followed by a newline or return carriage or a quotation mark
	"""
	return len(re.findall(r"[.,!?](?!\s)[^\"\n\r\']", text))


def count_extra_leading_whitespaces(text):
	"""
	find extra whitespace before `.,!?:;`
	"""
	return len(re.findall(r"\s[.,!?\:;]", text))


def count_missing_leading_whitespaces(text):
	return len(re.findall(r"(?<!\s)[\(]", text))


def count_missing_paired_characters(text, paired_chars=("()", '""', "''")):
	"""
	this only count open vs closed and does not check whether the order is correct
	"""
	return np.sum([np.abs(text.count(chars[0]) - text.count(chars[1])) for chars in paired_chars])


def squeeze_pattern(text, pattern="\n"):
	while pattern * 2 in text:
		text = text.replace(pattern * 2, pattern)
	return text


def clean_special_characters(text):
	cleaned_text = squeeze_pattern(text, pattern="\n")
	return cleaned_text.replace("\r", "").replace("\n", " ")


def number_of_unigrams(series: pd.Series) -> np.array:
	return series.apply(clean_special_characters).str.split(" ").apply(len).values.reshape(-1, 1)


def number_of_line_breaks(series: pd.Series) -> np.array:
	return series.str.count("\n").values.reshape(-1, 1)


def _get_punctuation_error_fraction(text):
	"""
	A definitely non exhaustive and imperfect list of punctuation errors.
	Can be higher than one as there can be more than one error per character
	"""
	error = (
		count_missing_trailing_whitespaces(text),
		count_extra_leading_whitespaces(text),
		count_missing_leading_whitespaces(text),
		count_missing_paired_characters(text)
	)
	total_punctuation_characters = count_punctuation_characters(text)
	if total_punctuation_characters > 0:
		return np.sum(error) / count_punctuation_characters(text)
	else:
		return 0


def get_punctuation_error_fraction(series: pd.Series) -> np.array:
	X_count = series.apply(count_punctuation_characters).values.reshape(-1, 1)
	X_fraction = series.apply(_get_punctuation_error_fraction).values.reshape(-1, 1)
	return np.concatenate((X_count, X_fraction), axis=1)

