import src.utils.json_utils as json_utils
from src.config.file_constants import MODEL_DESIGN

class HPOOptimizerSet:
    """
    A class to represent the base functionality for Hyperparameter Optimization (HPO) frameworks.

    Attributes:
        model_design (dict): A dictionary containing the model design configuration loaded 
                             from a JSON file specified by the `MODEL_DESIGN` constant.
    
    Methods:
        get_layer_params():
            Retrieves the keys representing the layer level model parameters in the search space.
            If a 'focus' key exists in the model design, only those parameters will be optimized alongside model level parameters 
            like "batch_size", "learning_rate" and "optimizer". Otherwise, returns all layer level model paramters.
    """

    def __init__(self):
        """
        Initializes the HPOOptimizerSet instance by loading the model design configuration.

        Raises:
            FileNotFoundError: If the JSON file specified by `MODEL_DESIGN` is not found.
            JSONDecodeError: If the JSON file contains invalid data.
        """
        self.model_design = json_utils.read_json_file(MODEL_DESIGN)
    
    def get_layer_params(self):
        """
        Get the keys in the search space that are layer level model parameters.

        If the `model_design` dictionary contains a 'focus' key, its associated with layer level parameters we want to optimize.
        The point is to optimize only some aspects of the model like `neurons` or `network type` and not all parameters. 
        Otherwise, it returns all layer related parameters.

        Returns:
            set: A set of default parameter keys to be processed.
        """
        if 'focus' in self.model_design:
            return self.model_design['focus']
        
        # Assuming `search_space` is defined in the child class
        return ['layers', 'neurons', 'dropout_rate', 'activation']
