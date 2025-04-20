class ModelMetrics:
    """
    A class to store and manage evaluation metrics and training details.

    Attributes:
        training_time (float): Total time taken to train the model (in seconds).
        epochs (int): Total number of epochs used for training.
        mae (float): Mean Absolute Error of the model predictions.
        mse (float): Mean Squared Error of the model predictions.
        smape (float): Symmetric Mean Absolute Percentage Error of the model predictions.
        rmse (float): Root Mean Squared Error of the model predictions.
        loss (float): Final training loss.
        val_loss (float): Final validation loss.
        aunl (float): Area Under Normalized Loss for training.
        aunl_val (float): Area Under Normalized Loss for validation.
    """

    def __init__(self):
        """
        Initializes the ModelMetrics object with default values for all attributes.
        """
        self.training_time = 0
        self.epoch = 0
        self.mae = 0
        self.mse = 0
        self.smape = 0
        self.rmse = 0
        self.loss = 0
        self.val_loss = 0
        self.aunl = 0
        self.aunl_val = 0

    def update(self, set):
        """
        Updates the metrics and training details for the model.

        Args:
            set (dict): Dictionary of all variables.
        """
        # If params is provided, update the class attributes
        if set:
            for key, value in set.items():
                if hasattr(self, key):  # Only update attributes that exist
                    setattr(self, key, value)

    def to_dict(self):
        """
        Converts the model metrics into a dictionary.

        Returns:
            dict: Dictionary containing all model metric attributes and their values.
        """
        return {attr: getattr(self, attr) for attr in vars(self)}