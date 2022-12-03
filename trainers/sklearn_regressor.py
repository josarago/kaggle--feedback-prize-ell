import xgboost as xgb

from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from trainers.base_trainer import (
	SCORE_COLUMNS,
	FEATURE_COLUMNS,
	ModelTrainer
)

from config import MSFTDeBertaV3Config, DEFAULT_DEBERTA_CONFIG


class SklearnRegressorTrainer(ModelTrainer):

	def __init__(
			self,
			model_type,
			deberta_config: MSFTDeBertaV3Config = DEFAULT_DEBERTA_CONFIG,
			batch_inference=True,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS,
			train_file_name=None,
			test_file_name=None,
			submission_filename=None,
	):
		super().__init__(
			deberta_config,
			batch_inference=batch_inference,
			target_columns=target_columns,
			feature_columns=feature_columns,
			train_file_name=train_file_name,
			test_file_name=test_file_name,
			submission_filename=submission_filename,
		)
		self._model_type = model_type

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

