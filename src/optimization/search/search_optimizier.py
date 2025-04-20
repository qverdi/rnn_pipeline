from pathlib import Path
import optuna
from optuna.samplers import RandomSampler, TPESampler
from src.config.file_constants import OUTPUT_DIR
from src.experiments.experiment_tracker import ExperimentTracker
from src.optimization.hpo_stopping import HPOStopping
from src.optimization.hpo_termination_error import HPOTerminationError
from src.experiments.experiment_param import ExperimentParam
from src.optimization.search.search_optimizer_callback import SearchStopCallback
from src.optimization.search.search_optimizer_set import SearchOptimizerSet
from src.models.model_training import ModelTraining
from typing import Dict, Any
from src.utils.log_write import LogWriter


class SearchOptimizer:
    """
    Implements hyperparameter optimization using TPE or Random Search.

    The optimizer adjusts hyperparameters within a specified search space to minimize
    the validation loss of a model. It supports budget-aware termination and dynamic
    parameter decoding.
    
    Attributes:
        search_space (dict): Defines the hyperparameters and their ranges for optimization.
        experiment (ExperimentParam): Contains details about the experiment setup, 
            including optimizer configurations, patience, and budget constraints.
        n_trials (int): Defines the number of itearions.
        sampler (OptunaSampler): Defines sampling method for optimization (TPE, Random).        
        tracker (ExperimentTracker): Tracks metrics for optimization progress and best results.
        set_creator (SearchOptimizerSet): Utility to generate parameter sets from optimization vectors.
    """       
    
    def __init__(self, search_space: Dict[str, Any], experiment: ExperimentParam):
        """
        Initializes the SearchOptimizer with the search space and experiment parameters.

        Args:
            search_space (dict): The hyperparameter search space to explore.
            experiment (ExperimentParam): The experiment parameters to guide the optimization.
        """
        self.search_space = search_space
        self.experiment = experiment

        self.tracker = ExperimentTracker()
        self.tracker.budget.value = self.experiment.budget_value
        
        output_file = OUTPUT_DIR / f"e_{self.experiment.id}_{Path(self.experiment.file_path).stem}.csv"
        self.writer = LogWriter(output_file, self.tracker.log_queue)

        self.__init_framework()
        
        if self.experiment.include_budget:
            self.__init_budget()

    def __init_framework(self):
        """
        Initializes the optimization framework, including setting the number of trials,
        selecting the appropriate sampler, and setting up the optimizer.
        """
        self.n_trials = self.experiment.optimizer_params.trials
        self.sampler = self.load_sampler(self.experiment.optimizer_params.sampler)
        self.set_creator = SearchOptimizerSet(self.search_space, self.experiment)
        self.callbacks = []

    def __init_budget(self):
        """
        Initializes budget-related settings and adds stopping criteria callbacks if the budget is included in the experiment.
        """
        self.tracker.comparison_metric = self.experiment.reference_metric       
        if self.experiment.include_budget:
            self.callbacks.append(SearchStopCallback(HPOStopping(self.experiment, self.tracker)))

    def optimize(self):
        """
        Starts the optimization process by creating an Optuna study and running the optimization.

        This function uses the study to optimize the `objective` function over a number of trials.
        It also handlesearly termination of the optimization process due to budget exhaustion.

        The best model's performance metrics (AUNL and the selected metric) are printed after the optimization.
        """
        try:
            # Create Optuna study and start optimization
            pruner = self.load_pruning() if self.experiment.sh else None
            study = optuna.create_study(direction="minimize", sampler=self.sampler, pruner=pruner)
            study.optimize(self.objective, n_trials=self.n_trials, callbacks=self.callbacks, n_jobs=self.experiment.optimizer_params.n_jobs)
        except HPOTerminationError as e: # Budget related termination
            # Handle termination of study
            print(f"Study terminated: {e}")

        # Print the best results from the optimization
        print(f"Best AUNL: {self.tracker.best_aunl.value}")
        print(f"Best {self.tracker.comparison_metric}: {self.tracker.best_value.value}")
        
        if self.experiment.sh:
            print("Pruned trials:", len(study.get_trials(states=[optuna.trial.TrialState.PRUNED])))

        self.writer.stop()
    

    def objective(self, trial: optuna.Trial) -> float:
        """
        The objective function for Optuna's optimization. This function trains the model 
        and returns the validation loss as the objective to minimize.

        Args:
            trial (optuna.Trial): The trial object representing a single optimization iteration.

        Returns:
            float: The validation loss of the trained model for the current set of hyperparameters.
        """
        # Create parameter set from the trial's suggested hyperparameters
        parameter_set = self.set_creator.create_model_params(trial)

        # Train the model with the current hyperparameters and get the evaluation metrics
        model_training = ModelTraining(self.experiment, self.tracker, parameter_set, {'trial': trial})
        objective_metric = model_training.train_model()

        # Return the validation loss, which is the value to minimize
        return objective_metric

    def load_sampler(self, sampler: str) -> optuna.samplers.BaseSampler:
        """
        Loads the appropriate Optuna sampler based on the specified method.

        Args:
            sampler (str): The name of the sampler to load. Can be "random" or "bayesian".

        Returns:
            optuna.samplers.BaseSampler: The loaded sampler object.

        Raises:
            ValueError: If an unsupported sampler name is provided.
        """
        # Select and return the corresponding Optuna sampler
        if sampler == "random":
            return RandomSampler()
        elif sampler == "bayesian":
            return TPESampler()
        else:
            raise ValueError(f"Unsupported sampler: {sampler}")
        
    
    def load_pruning(self):
        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=self.experiment.patience,
            reduction_factor=2,
            min_early_stopping_rate=0
        )

        return pruner

