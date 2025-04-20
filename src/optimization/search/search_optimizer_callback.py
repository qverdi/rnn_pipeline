from src.optimization.hpo_stopping import HPOStopping
from src.optimization.hpo_termination_error import HPOTerminationError
import optuna


class SearchStopCallback:
    def __init__(self, criteria: HPOStopping):
        """
        Initializes the SearchStopCallback with a given stopping criteria.

        Args:
            criteria (HPOStopping): An instance of HPOStopping that defines the stopping conditions.
        """
        self.criteria = criteria

    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        """
        Check whether the stopping criteria are met and, if so, raise an exception to terminate the optimization process.

        Args:
            study (optuna.study.Study): The current Optuna study object.
            trial (optuna.trial.FrozenTrial): The current trial being evaluated.

        Raises:
            HPOTerminationError: If the stopping criteria are met, the optimization process is halted by raising an exception.
        """
        if self.criteria.should_terminate():
            # If the stopping criteria is met, terminate the optimization with an exception
            raise HPOTerminationError("Optimization halted by stopping criteria.")
