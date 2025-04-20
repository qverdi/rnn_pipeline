import src.utils.json_utils as json_utils
from src.config.file_constants import MODEL_OPTIMIZER_PARAMS
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

class ModelOptimizerParams:
    """
    A class to create optimizer parameters for deep learning models.

    Attributes:
        beta_1 (float): Exponential decay rate for the first moment estimates (Adam optimizer).
        beta_2 (float): Exponential decay rate for the second moment estimates (Adam optimizer).
        amsgrad (bool): Whether to apply the AMSGrad variant of Adam.
        weight_decay (float): Weight decay (L2 penalty) for regularization.
        clipnorm (float): Maximum norm for gradient clipping.
        clipvalue (float): Maximum value for gradient clipping.
        clipping (str): Type of gradient clipping ("clipnorm", "clipvalue").
        rho (float): Decay factor for RMSprop optimizer.
        momentum (float): Momentum factor for SGD and RMSprop optimizers.
        centered (bool): Whether to normalize gradients (RMSprop).
        nesterov (bool): Whether to apply Nesterov momentum (SGD).
        optimizer (str): The selected optimizer type.
    """

    def __init__(self, params: dict):
        """
        Initializes the ModelOptimizerParams with given parameters.

        Args:
            params (dict): Dictionary of optimizer parameters.
        """
        self.beta_1 = 0.9
        self.beta_2 = 0.999
        self.amsgrad = False
        self.weight_decay = None
        self.clipnorm = None
        self.clipvalue = None
        self.clipping = None
        self.rho = 0.9
        self.momentum = 0.0
        self.centered = False
        self.nesterov = False

        if params:
            for key, value in params.items():
                if hasattr(self, key):  # Ensure the attribute exists before updating
                    setattr(self, key, value)
        
        self.optimizer = params["optimizer"]
        self.__define_clipping()

    @staticmethod
    def get_all_keys():
        """
        Retrieves all available optimizer parameter keys from a JSON configuration file.

        Returns:
            list: A list of all available optimizer parameter keys.
        """
        return list(json_utils.read_json_file(MODEL_OPTIMIZER_PARAMS).keys())

    @staticmethod
    def get_params():
        """
        Retrieves optimizer parameters from a JSON configuration file.

        Returns:
            dict: Dictionary containing optimizer parameters.
        """
        return json_utils.read_json_file(MODEL_OPTIMIZER_PARAMS)

    def __define_clipping(self):
        """
        Defines the type of gradient clipping to be used based on the clipping strategy.
        """
        self.clipnorm = self.clipnorm if self.clipping == "clipnorm" else None
        self.clipvalue = self.clipvalue if self.clipping == "clipvalue" else None
    
    def get_optimizer(self, learning_rate):
        """
        Returns the optimizer instance based on the current parameter set.

        Args:
            learning_rate (float): The learning rate to be used by the optimizer.

        Returns:
            tensorflow.keras.optimizers.Optimizer: An instance of the selected optimizer.
        """      
        optimizers = {
            "adam": Adam(
                learning_rate=learning_rate,
                beta_1=self.beta_1,
                beta_2=self.beta_2,
                amsgrad=self.amsgrad,
                weight_decay=self.weight_decay,
                clipnorm=self.clipnorm,
                clipvalue=self.clipvalue,
            ),
            "sgd": SGD(
                learning_rate=learning_rate,
                momentum=self.momentum,
                nesterov=self.nesterov,
                weight_decay=self.weight_decay,
                clipnorm=self.clipnorm,
                clipvalue=self.clipvalue,
            ),
            "rmsprop": RMSprop(
                learning_rate=learning_rate,
                rho=self.rho,
                momentum=self.momentum,
                centered=self.centered,
                weight_decay=self.weight_decay,
                clipnorm=self.clipnorm,
                clipvalue=self.clipvalue,
            ),
        }
        if self.optimizer not in optimizers:
            raise ValueError(f"Unsupported optimizer: {self.optimizer}")
        return optimizers[self.optimizer]
    
    def __str__(self):
        """
        Returns a formatted string representation of the object's attributes.

        Returns:
            str: A string representation of the model parameters.
        """
        attributes = vars(self)  # Get all attributes as a dictionary
        return "\n".join(f"{key}: {value}" for key, value in attributes.items())
