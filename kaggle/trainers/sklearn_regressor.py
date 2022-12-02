

from .base_trainer import ModelTrainer


class SklearnRegressorTrainer(ModelTrainer):
	def __init__(
			self,
			deberta_config: MSFTDeBertaV3Config,
			model_type,
			fastext_model_path,
			batch_inference=True,
			train_filename=TRAIN_FILENAME,
			test_filename=TEST_FILENAME,
			submission_filename=SUBMISSION_FILENAME,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS,
	):

		assert model_type in ("dummy", "linear", "xgb"), "'model_type' must be either 'xgb', 'linear' or 'dummy'"
		self._deberta_config = deberta_config
		self._batch_inference = batch_inference
		self._model_type = model_type
		self._train_filename = train_filename
		self._test_filename = test_filename
		self._submission_filename = submission_filename
		self._target_columns = target_columns
		self._feature_columns = feature_columns
		self._model = None

	@classmethod
	def from_file(
			cls,
			deberta_config,
			train_filename,
			model_type,
			batch_inference,
			target_columns=SCORE_COLUMNS
	):
		if os.path.exists(train_filename):
			return cls(deberta_config, model_type, batch_inference=batch_inference, train_filename=train_filename)
		else:
			raise ValueError(f"file '{train_filename}' does not exist.")

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
