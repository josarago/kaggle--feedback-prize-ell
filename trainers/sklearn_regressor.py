import xgboost as xgb
import lightgbm as lgb

from sklearn.multioutput import MultiOutputRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA
from sklearn.compose import TransformedTargetRegressor

from trainers.base_trainer import (
	SCORE_COLUMNS,
	FEATURE_COLUMNS,
	ModelTrainer
)

from config import MSFTDeBertaV3Config


class SklearnRegressorTrainer(ModelTrainer):

	def __init__(
			self,
			model_type,
			fastext_model_path,
			deberta_config: MSFTDeBertaV3Config,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS,
			train_file_name=None,
			test_file_name=None,
			submission_filename=None,
	):
		super().__init__(
			fastext_model_path,
			deberta_config,
			target_columns=target_columns,
			feature_columns=feature_columns,
			train_file_name=train_file_name,
			test_file_name=test_file_name,
			submission_filename=submission_filename,
		)
		self._model_type = model_type

	def train(self, X, y, params=None, pca_on_target=True):
		if self._model_type == "lgb":
			print("creating LightGBM regressor")
			_model = MultiOutputRegressor(lgb.LGBMRegressor(**params if params else {}))
		elif self._model_type == "xgb":
			print("creating XGBoost regressor")
			_model = MultiOutputRegressor(xgb.XGBRegressor(**params if params else {}))
		elif self._model_type == "linear":
			print("creating linear model")
			_model = LinearRegression()
		elif self._model_type == "dummy":
			print("creating dummy model")
			_model = DummyRegressor(strategy="mean")
		else:
			raise ValueError("unknown model type")
		if pca_on_target:
			pca = PCA(n_components=y.shape[1])
			pca.fit(y)
			self._model = TransformedTargetRegressor(
				regressor=_model,
				func=pca.inverse_transform,
				inverse_func=pca.transform
			)
		else:
			self._model = _model
		self._model.fit(X, y)

	def predict(self, X, recast_scores=True):
		y_pred = self._model.predict(X)
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred

