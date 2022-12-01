import os
import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error

from config import MSFTDeBertaV3Config


KAGGLE_ROOT_DIR = "/kaggle"
INPUT_DIR = "input"
CHALLENGE_NAME = "feedback-prize-english-language-learning"
SUBMISSION_DIR = "working"

TEST_SIZE = 0.33

PARAMS = dict()
PARAMS["dummy"] = dict()
PARAMS["linear"] = dict()
PARAMS["xgb"] = dict(
    booster='gbtree',
    colsample_bylevel=1.0,
    colsample_bytree=1.0,
#     early_stopping_rounds=10,
#     external_memory=False,
    gamma=0.0,
    learning_rate=0.1,
    max_delta_step=0.0,
    max_depth=7,
    in_child_weights=1.0,
    n_estimators=50, #100
    normalize_type='tree',
    num_parallel_tree=1,
    n_jobs=1,
    objective='reg:squarederror',
    one_drop=False,
    rate_drop=0.0,
    reg_alpha=0.0,
    reg_lambda=1.0,
    sample_type='uniform',
    silent=True,
    skip_drop=0.0,
    subsample=1.0
)


PARAMS["lgbm"] = dict(
    boosting_type='gbdt',
    num_leaves=15, #31
    max_depth=14, #-1
    learning_rate=0.05,
    n_estimators=50, #100
    subsample_for_bin=200000,
    objective=None,
    min_split_gain=0.1,
    min_child_weight=0.001,
    min_child_samples=10,
    subsample=0.9, #1.0
    subsample_freq=0,
    colsample_bytree=1.0,
    reg_alpha=0.0,
    reg_lambda=0.0,
    random_state=None,
    n_jobs=None,
    importance_type='split',
#     verbosity=1
)

BATCH_SIZE = 512

PARAMS["nn"] = dict(
    hidden_dims=[6],
    n_hidden = None,
    batch_size=BATCH_SIZE,
    force_half_points=False,
    num_epochs = 500,
    learning_rate = 0.0001,
    shuffle=True,
    val_size=0.25,
    with_validation=True,
    device="cpu"
)


def get_file_names():
	""""
		generates filenames used to train, test and submit.
		Works locally (assuming the correct directory structure) and on kaggle
	"""
	train_file_name = os.path.join(KAGGLE_ROOT_DIR, INPUT_DIR, CHALLENGE_NAME, "train.csv")
	test_file_name = os.path.join(KAGGLE_ROOT_DIR, INPUT_DIR, CHALLENGE_NAME, "test.csv")
	submission_filename = os.path.join(KAGGLE_ROOT_DIR, SUBMISSION_DIR, "submission.csv")
	return train_file_name, test_file_name, submission_filename


TRAIN_FILENAME, TEST_FILENAME, SUBMISSION_FILENAME = get_file_names()
print(TRAIN_FILENAME)



deberta_config = MSFTDeBertaV3Config(
	model_size="base",
	pooling="mean",
	inference_device="cuda:0",
	output_device="cuda:0",
	inference_batch_size=100
)




