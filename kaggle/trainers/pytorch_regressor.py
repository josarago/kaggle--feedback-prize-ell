










class NNTrainer(ModelTrainer):
	def __init__(
			self,
			deberta_config,
			device,
			fastext_model_path,
			batch_inference=True,
			train_filename=TRAIN_FILENAME,
			test_filename=TEST_FILENAME,
			submission_filename=SUBMISSION_FILENAME,
			target_columns=SCORE_COLUMNS,
			feature_columns=FEATURE_COLUMNS,
	):
		self._deberta_config = deberta_config
		self._batch_inference = batch_inference
		self._device = device
		self._train_filename = train_filename
		self._test_filename = test_filename
		self._submission_filename = submission_filename
		self._target_columns = target_columns
		self._feature_columns = feature_columns
		self._model = None
		self._optimizer = None
		self._loss_fn = nn.MSELoss()

	@classmethod
	def from_file(cls, deberta_config, device, batch_inference, train_filename, target_columns=SCORE_COLUMNS):
		if os.path.exists(train_filename):
			return cls(deberta_config, device, batch_inference=batch_inference, train_filename=train_filename)
		else:
			raise ValueError(f"file '{train_filename}' does not exist.")

	def get_data_loader(self, X, y, bactch_size, shuffle=True):
		data = Data(X, y)
		data_loader = DataLoader(
			dataset=data,
			batch_size=bactch_size,
			shuffle=shuffle,
			collate_fn=lambda x: tuple(x_.to(self._device) for x_ in default_collate(x))
		)
		return data_loader

	def train(self, X, y, params):
		self._model = SequentialNeuralNetwork(
			X,
			y,
			hidden_dims=params["hidden_dims"],
			n_hidden=params["n_hidden"],
			force_half_points=params["force_half_points"]
		)
		self._model.to(self._device)
		self._optimizer = torch.optim.Adam(self._model.parameters(), lr=params["learning_rate"])
		self._loss_values = dict()
		self._loss_values["train"] = []
		if params["with_validation"]:
			print("Using validation")
			self._loss_values["val"] = []
			X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=params["val_size"])
			train_data_loader = self.get_data_loader(X_train, y_train, params["batch_size"], params["shuffle"])
			val_data_loader = self.get_data_loader(X_val, y_val, params["batch_size"], params["shuffle"])
		else:
			train_data_loader = self.get_data_loader(X, y, params["batch_size"], params["shuffle"])

		min_val_loss = np.inf

		for epoch in range(params["num_epochs"]):
			_loss_values = dict(train=[])

			for X_train, y_train in train_data_loader:
				# Compute prediction error
				y_pred_train = self._model(X_train)
				train_loss = self._loss_fn(y_pred_train, y_train)
				_loss_values["train"].append(train_loss.item())

				# Backpropagation
				self._optimizer.zero_grad()
				train_loss.backward()
				self._optimizer.step()

			self._loss_values["train"].append(np.mean(_loss_values["train"]))
			if params["with_validation"]:
				_loss_values["val"] = []
				self._model.eval()  # Optional when not using Model Specific layer
				for X_val, y_val in val_data_loader:
					# Forward Pass
					y_pred_val = self._model(X_val)
					# Find the Loss
					val_loss = self._loss_fn(y_pred_val, y_val)
					# Calculate Loss
					_loss_values["val"].append(val_loss.item())
			self._loss_values["val"].append(np.mean(_loss_values["val"]))

		print("Training Complete")

	def plot_loss_values(self):
		fig, ax = plt.subplots(figsize=(20, 5))
		ax.plot(np.array(self._loss_values["train"]), label="training")
		if "val" in self._loss_values.keys():
			ax.plot(np.array(self._loss_values["val"]), label="validation")
		ax.set_xlabel("Epochs")
		ax.set_ylabel("Losses")
		ax.legend()
		plt.show()

	def predict(self, X, recast_scores=True):
		y_pred = self._model(Data.csr_to_torch(X).to(device)).cpu().detach().numpy()
		if recast_scores:
			y_pred = self.recast_scores(y_pred)
		return y_pred

	def make_submission_df(self, recast_scores=True, write_file=True):
		"""
			implement batch prediction to avoid OOM errors?
		"""
		print(f"loading test file from : '{self._test_filename}'")
		submission_df = pd.read_csv(self._test_filename)
		X_submission = self._pipeline.transform(submission_df)
		y_pred_submission = self.predict(X_submission, recast_scores=recast_scores)

		submission_df[self._target_columns] = y_pred_submission
		submission_df = submission_df[["text_id"] + self._target_columns]
		if write_file:
			print(f"Writing submission to: '{self._submission_filename}'")
			submission_df.to_csv(self._submission_filename, index=False)
		return submission_df
