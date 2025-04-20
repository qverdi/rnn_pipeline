from src.experiments.experiment_param import ExperimentParam
from src.experiments.experiment_tracker import ExperimentTracker


class HPOStopping:
    """
    The HPOStopping class provides stopping criteria for hyperparameter optimization (HPO) based on the available budget
    and the optimization strategy used. It supports both fixed and greedy stopping strategies.

    Stopping criteria include:
    - **Fixed Stopping**: Stops optimization when the budget is exhausted.
    - **Greedy Stopping**: Applies different stopping rules depending on the optimizer used (e.g., search-based or generation-based optimizers).
    Greedy stopping also considers whether the remaining budget is sufficient for continuing the optimization process.

    Args:
        experiment (ExperimentParam): The experiment parameters to guide the stopping criteria.
    """    
    def __init__(self, experiment: ExperimentParam, tracker: ExperimentTracker):
        """
        Initializes the HPOStopping class with the given experiment parameters.

        Args:
            experiment (ExperimentParam): The experiment parameters to guide the stopping criteria.
        """
        self.experiment = experiment
        self.strategy = self.experiment.budget_strategy
        self.tracker = tracker

    def should_terminate(self, estimated_epochs=None):
        """
        Determines whether the optimization process should terminate based on the budget strategy.

        Args:
            estimated_epochs (int, optional): The estimated number of epochs, used in the "greedy" strategy.

        Returns:
            bool: Whether the optimization should stop based on the strategy.
        """
        # Check stopping condition based on the budget strategy (fixed or greedy)
        if self.strategy == "fixed":
            return self.fixed_stopping()
        if self.strategy == "greedy":
            return self.greedy_stopping(estimated_epochs)
        return False

    def fixed_stopping(self):
        """
        Determines if the optimization should stop based on a fixed budget condition.

        Returns:
            bool: Whether the optimization should stop based on a fixed budget (budget <= 0).
        """
        # Fixed stopping criteria: if the budget is exhausted, stop
        with self.tracker.budget.get_lock():
            criteria = self.tracker.budget.value <= 0
            if criteria:
                print("Budget = 0. Fixed stopping.")
            return criteria

    def greedy_stopping(self, estimated_epochs=None):
        """
        Determines if the optimization should stop based on a greedy strategy.

        Args:
            estimated_epochs (int, optional): The estimated number of epochs, used in some greedy strategies.

        Returns:
            bool: Whether the optimization should stop based on a greedy stopping strategy.
        """
        with self.tracker.budget.get_lock():
            return self.greedy_search_stopping()

    def greedy_search_stopping(self):
        """
        Greedy stopping criteria for search-based optimizers. Model-level checking if optimization should terminate.

        Returns:
            bool: Whether the optimization should stop based on the search greedy strategy.
        """
        # If the budget is exhausted, stop the optimization
        if self.tracker.budget.value <= 0:
            print("Budget exceeded. Greedy stopping.")
            return True
        # Check if the budget with tolerance is smaller than half of the total epochs
        if self.tracker.budget.value < self.experiment.budget_tolerance:
            criteria = (
                self.experiment.num_epochs / 2
            ) > self.tracker.budget.value + self.experiment.budget_tolerance
            if criteria:
                print("Budget exceeded. Greedy stopping.")
                print(f"CRITERIA FOR STOPPING {criteria}")
            return criteria
        return False