from typing import Dict
from src.optimization.search.search_optimizier import SearchOptimizer
from src.experiments.experiment_param import ExperimentParam


class HPOManager:
    """
    A class to manage hyperparameter optimization (HPO) using different optimization algorithms.

    This class selects an optimizer based on the `optimizer` attribute in the `experiment` parameter
    and runs the corresponding optimization algorithm on the provided search space.

    Attributes:
        optimizer (str): The type of optimizer to use ("search").
        search_space (dict): A dictionary defining the search space for hyperparameters.
        optimizer_instance (SearchOptimizer):
            The instance of the selected optimizer to perform the optimization.

    Methods:
        __init__(self, search_space: dict, experiment: ExperimentParam):
            Initializes the HPOManager with the search space and experiment parameters.

        optimize(self) -> None:
            Starts the optimization process using the selected optimizer.
    """

    def __init__(self, search_space: Dict[str, any], experiment: ExperimentParam):
        """
        Initializes the HPOManager with the given search space and experiment.

        Args:
            search_space (dict): A dictionary representing the hyperparameter search space.
            experiment (ExperimentParam): An instance of the ExperimentParam class containing the experiment configuration.

        Raises:
            ValueError: If the specified optimizer is unsupported.
        """
        self.optimizer = experiment.optimizer
        self.search_space = search_space

        # Select optimizer based on the 'optimizer' attribute of the experiment
        
        self.optimizer_instance = SearchOptimizer(self.search_space, experiment)
        
    def optimize(self) -> None:
        """
        Runs the optimization process using the selected optimizer.

        This method invokes the `optimize` method on the chosen optimizer instance.
        """
        self.optimizer_instance.optimize()
