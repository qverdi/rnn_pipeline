from src.models.model_optimizer_params import ModelOptimizerParams

class ModelParams:
    """
    A class to define and manage the hyperparameters and configurations for a machine learning model.

    Attributes:
        num_layers (int): Number of layers in the model.
        layers (list of str): Types of layers used in the model.
        neurons (list of int): Number of neurons in each layer.
        dropout_rate (list of float): Dropout rates for each layer.
        activation (list of str): Activation functions used in each layer.
        experiment_id (str): Unique identifier for the experiment.
        input_shape (tuple): Shape of the input data.
        loss (str): Loss function used for training.
        optimizer (str): Optimizer used for training.
        learning_rate (float): Learning rate for the optimizer.
        batch_size (int): Batch size used during training.
    """

    def __init__(self, params: dict):
        """
        Initializes the ModelParams object with default values and updates them
        based on the provided parameters.

        Args:
            params (dict): A dictionary of parameters to update the default values.
        """
        # Default values for model parameters
        self.num_layers = 3  # Number of layers in the model
        self.layers = ["rnn", "rnn", "rnn"]  # Layer types
        self.neurons = [16, 32, 16]  # Number of neurons in each layer
        self.dropout_rate = [0.1, 0.2, 0.1]  # Dropout rates for each layer
        self.activation = [
            "relu",
            "relu",
            "relu",
        ]  # Activation functions for each layer

        self.experiment_id = "00000000"  # Default experiment identifier

        # Input data shape and training configurations
        self.input_shape = (30, 1)  # Shape of the input data
        self.loss = "mse"  # Loss function
        self.optimizer = "adam"  # Optimizer
        self.learning_rate = 0.01  # Learning rate
        self.batch_size = 16  # Batch size
        self.optimizer_params = None

        # Update default attributes with values from the `params` dictionary
        if params:
            for key, value in params.items():
                if hasattr(self, key):  # Ensure the attribute exists before updating
                    setattr(self, key, value)

        self.optimizer_params = ModelOptimizerParams(params)

    def __str__(self):
        """
        Returns a formatted string representation of the object's attributes.

        Returns:
            str: A string representation of the model parameters.
        """
        attributes = vars(self)  # Get all attributes as a dictionary
        optimizers = self.optimizer_params.__str__()
        return "\n".join(f"{key}: {value}" for key, value in attributes.items()) + f"\nOptimizers: {optimizers}"
