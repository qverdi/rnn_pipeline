from src.optimization.search.search_optimizer_params import SearchOptimizerParams
from src.config.file_constants import (
    SEARCH_OPTIMIZER_PARAMS,
)
from src.utils import json_utils
import uuid
import hashlib


class ExperimentParam:
    """
    Represents the parameters for a single experiment. This class is responsible for initializing
    and managing experiment-specific configurations, including optimizer parameters.
    """

    def __init__(self, params: dict, file_path: str):
        """
        Initializes the experiment parameters with default values, updates them with provided `params`,
        and initializes the optimizer-specific parameters.

        Args:
            params (dict): A dictionary of experiment-specific parameters.
            file_path (str): The file path associated with the dataset for this experiment.
        """
        # Default parameters
        self.id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()[
            :8
        ]  # Generate a unique ID for each experiment
        self.file_path = file_path
        self.gpu = 0
        self.loss_function = "mse"
        self.num_epochs = 100
        self.optimizer = "search"
        self.objective = "singular_loss"
        self.objective_weights = [0.5, 0.5]
        self.early_stopping = "none"
        self.reference_metric = "mse"
        self.weight_sharing = False
        self.sh = False
        self.patience = 10
        self.train_size = 0.7
        self.step_check = 5
        self.val_size = 0.1
        self.target_column = 0
        self.window_size = 5
        self.optimizer_finetuning = False
        self.has_model = False
        self.is_descrete = False
        self.include_budget = False
        self.budget_value = 100
        self.budget_strategy = "fixed"  # Budget allocation strategy
        self.budget_tolerance = 1000  # Allowed tolerance for stopping
        
        self.scalers = {}

        # If params is provided, update the class attributes
        if params:
            for key, value in params.items():
                if hasattr(self, key):  # Only update attributes that exist
                    setattr(self, key, value)

        self.optimizer_params = self.__init_optimizer()

    def set_file_path(self, file_path: str):
        """
        Sets the file path for the experiment.

        Args:
            file_path (str): The new file path to set.
        """
        self.file_path = file_path

    def __init_optimizer(self):
        """
        Initializes the optimizer parameters based on the selected optimizer type.

        Returns:
            object: An instance of the corresponding optimizer parameter class.

        Raises:
            ValueError: If the optimizer type is unsupported.
        """
        optimizer_params = json_utils.read_json_file(SEARCH_OPTIMIZER_PARAMS)
        return SearchOptimizerParams(optimizer_params)


    def __str__(self):
        """
        Returns a string representation of the experiment parameters.

        Returns:
            str: A formatted string containing the experiment parameters.
        """
        return (
            f"Experiment Parameters:\n"
            f"  ID: {self.id}\n"
            f"  GPU Usage: {self.gpu}\n"
            f"  File Path: {self.file_path}\n"
            f"  Loss Function: {self.loss_function}\n"
            f"  Number of Epochs: {self.num_epochs}\n"
            f"  Optimizer: {self.optimizer}\n"
            f"  Patience: {self.patience}\n"
            f"  Train Size: {self.train_size}\n"
            f"  Validation Size: {self.val_size}\n"
            f"  Step Check: {self.step_check}\n"
            f"  Value Column: {self.target_column}\n"
            f"  Window Size: {self.window_size}\n"
        )
