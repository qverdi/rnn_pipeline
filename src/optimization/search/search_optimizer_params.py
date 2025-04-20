from typing import Dict


class SearchOptimizerParams:
    """
    A class to manage the parameters for a search optimization experiment.

    This class allows for configuring the search optimization parameters, such as the sampling method
    and the number of trials. It also allows for updating attributes based on a provided dictionary of parameters.

    Attributes:
        sampler (str): The method used for sampling during the search optimization (default is "random").
        trials (int): The number of trials to run during the search optimization (default is 20).

    Methods:
        __init__(self, params: dict):
            Initializes the SearchOptimizerParams with the given parameters.

        __str__(self) -> str:
            Returns a string representation of the search optimization parameters.
    """

    def __init__(self, params: Dict[str, any]):
        """
        Initializes the SearchOptimizerParams with the given parameters.

        Args:
            params (dict): A dictionary of parameters to configure the search optimization.
                           If a parameter key exists in the class, its value will be updated.

        Attributes set:
            sampler (str): Defines the search method (default is "random").
            trials (int): The number of trials to run (default is 20).
        """
        self.sampler = "random"
        self.trials = 20
        self.n_jobs = 1

        # If params is provided, update the class attributes
        if params:
            for key, value in params.items():
                if hasattr(self, key):  # Only update attributes that exist
                    setattr(self, key, value)

    def __str__(self) -> str:
        """
        Returns a string representation of the search optimization parameters.

        Returns:
            str: A formatted string describing the number of trials and the search space algorithm.
        """
        return (
            f"Experiment Search Parameters:\n"
            f"  Number of trials: {self.trials}\n"
            f"  Search space algorithm: {self.sampler}\n"
        )
