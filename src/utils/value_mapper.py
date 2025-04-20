from src.experiments.experiment_param import ExperimentParam
from src.experiments.experiment_tracker import ExperimentTracker
from src.models.model_params import ModelParams
import os


class ValueMapper:
    """
    A class to map values from one dimension to another.
    Converts normalized to denormalized value.
    Converts result classes to one final dictionary.

    Attributes:
        search_space (dict): A dictionary containing the search space for hyperparameters.
        is_descrete (bool): A flag to indicate if the mapping should be treated as discrete.
        categorical_to_range (dict): A dictionary mapping categorical parameters to their ranges.
        int_ranges (dict): A dictionary defining the integer ranges for certain parameters.
        continuous_ranges (dict): A dictionary defining the continuous ranges for certain parameters.
    """

    def __init__(self, search_space, is_descrete=False):
        """
        Initializes the ValueMapper with a search space and an optional flag for discrete mapping.

        Args:
            search_space (dict): The dictionary containing the search space for hyperparameters.
            is_descrete (bool, optional): If True, the mapping is treated as discrete. Defaults to False.
        """
        self.search_space = search_space
        self.is_descrete = is_descrete

        # Initialize mappings for categorical parameters with their ranges
        self.categorical_to_range = {
            "optimizer": search_space["optimizer"],
            "activation": search_space["activation"],
            "layers": search_space["layers"],
        }

        # Define integer ranges for integer parameters
        self.int_ranges = {
            "batch_size": search_space["batch_size"],
            "num_layers": search_space["num_layers"],
            "neurons": search_space["neurons"],
        }

        # Define continuous ranges for continuous parameters
        self.continuous_ranges = {
            "dropout_rate": search_space["dropout_rate"],
            "learning_rate": search_space["learning_rate"],
        }

    def update_search_space(self, addition, search_space):
        if addition == "model_optimizer":
            self.categorical_to_range.update(
                {
                    "centered": search_space["centered"],
                    "nesterov": search_space["nesterov"],
                    "amsgrad": search_space["amsgrad"],
                    "clipping": search_space["clipping"],
                }
            )

            self.continuous_ranges.update(
                {
                    "beta_1": search_space["beta_1"],
                    "beta_2": search_space["beta_2"],
                    "rho": search_space["rho"],
                    "momentum": search_space["momentum"],
                    "weight_decay": search_space["weight_decay"],
                    "clipnorm": search_space["clipnorm"],
                    "clipvalue": search_space["clipvalue"],
                }
            )

            self.search_space.update(search_space)

    def denormalize(self, value, param):
        """
        Denormalizes a given value based on the parameter type (categorical, integer, or continuous).

        Args:
            value (float): The normalized value to be denormalized.
            param (str): The parameter name to map the value back to its original scale.

        Returns:
            The denormalized value for the given parameter.

        Raises:
            ValueError: If the parameter is not recognized.
        """
        # Ensure value is between 0 and 1 (for categorical or continuous normalization)
        value = max(0, min(value, 1))

        if param in self.categorical_to_range:
            return self._denormalize_categorical(value, param)

        if param in self.int_ranges:
            return self._denormalize_int(value, param)

        if param in self.continuous_ranges:
            return self._denormalize_continuous(value, param)

        raise ValueError(f"Invalid parameter: {param}.")

    def _denormalize_categorical(self, value, param):
        """
        Denormalizes categorical values based on their normalized range.

        Args:
            value (float): The normalized value between 0 and 1.
            param (str): The categorical parameter to map the value.

        Returns:
            The original categorical value mapped from the normalized value.
        """
        index = int(value * len(self.search_space[param]))
        index = min(index, len(self.search_space[param]) - 1)
        return self.search_space[param][index]

    def _denormalize_int(self, value, param):
        """
        Denormalizes integer values within their defined range.

        Args:
            value (float): The normalized value between 0 and 1.
            param (str): The integer parameter to map the value.

        Returns:
            The denormalized integer value.
        """
        int_values = self.search_space[param]

        if self.is_descrete:
            return self._denormalize_categorical(value, param)

        range_min, range_max = min(int_values), max(int_values)
        return int(value * (range_max - range_min) + range_min)

    def _denormalize_continuous(self, value, param):
        """
        Denormalizes continuous values based on their defined range.

        Args:
            value (float): The normalized value between 0 and 1.
            param (str): The continuous parameter to map the value.

        Returns:
            The denormalized continuous value.
        """
        cont_values = self.search_space[param]

        if self.is_descrete:
            return self._denormalize_categorical(value, param)

        range_min, range_max = min(cont_values), max(cont_values)
        mapped_value = value * (range_max - range_min) + range_min
        return round(mapped_value, 4)

    @staticmethod
    def get_log_dict(
        experiment: ExperimentParam,
        tracker: ExperimentTracker,
        model_params: ModelParams,
        model_id: str,
    ):
        """
        Creates a dictionary containing experiment and model parameters for logging.

        Args:
            experiment (ExperimentParam): The experiment parameters to log.
            model_params (ModelParams): The model parameters to log.
            model_id (str): The ID of the model being logged.

        Returns:
            dict: A dictionary with the relevant parameters for logging.
        """
        row = {
            "experiment_id": experiment.id,
            "file_path": os.path.basename(experiment.file_path),
            "loss_function": experiment.loss_function,
            "num_epochs": experiment.num_epochs,
            "hpo": experiment.optimizer,
            "objective": experiment.objective,
            "objective_weights": experiment.objective_weights,
            "patience": experiment.patience,
            "train_size": experiment.train_size,
            "val_size": experiment.val_size,
            "step_check": experiment.step_check,
            "window_size": experiment.window_size,
            "optimizer_finetuning": experiment.optimizer_finetuning,
            "is_descrete": experiment.is_descrete,
            "has_model": experiment.has_model,
            "early_stopping": experiment.early_stopping,
            "reference_metric": experiment.reference_metric,
            "include_budget": experiment.include_budget,
            "budget": None,
            "budget_strategy": (
                experiment.budget_strategy if experiment.include_budget else None
            ),
            "budget_tolerance": (
                experiment.budget_tolerance if experiment.include_budget else None
            ),
            "n_trails": (
                experiment.optimizer_params.trials
                if experiment.optimizer == "search"
                else None
            ),
            "search_sampler": (
                experiment.optimizer_params.sampler
                if experiment.optimizer == "search"
                else None
            ),
            "n_jobs": experiment.optimizer_params.n_jobs,
            "model_id": model_id,
            "num_layers": model_params.num_layers,
            "layers": model_params.layers,
            "neurons": model_params.neurons,
            "activation": model_params.activation,
            "dropout_rate": model_params.dropout_rate,
            "learning_rate": model_params.learning_rate,
            "batch_size": model_params.batch_size,
            "optimizer": model_params.optimizer,
            "beta_1": (
                model_params.optimizer_params.beta_1
                if experiment.optimizer_finetuning and model_params.optimizer == "adam"
                else None
            ),
            "beta_2": (
                model_params.optimizer_params.beta_2
                if experiment.optimizer_finetuning and model_params.optimizer == "adam"
                else None
            ),
            "rho": (
                model_params.optimizer_params.rho
                if experiment.optimizer_finetuning
                and model_params.optimizer == "rmsprop"
                else None
            ),
            "momentum": (
                model_params.optimizer_params.momentum
                if experiment.optimizer_finetuning
                and (
                    model_params.optimizer == "rmsprop"
                    or model_params.optimizer == "sgd"
                )
                else None
            ),
            "centered": (
                model_params.optimizer_params.centered
                if experiment.optimizer_finetuning
                and model_params.optimizer == "rmsprop"
                else None
            ),
            "weight_decay": (
                model_params.optimizer_params.weight_decay
                if experiment.optimizer_finetuning
                else None
            ),
            "clipnorm": (
                model_params.optimizer_params.clipnorm
                if experiment.optimizer_finetuning
                else None
            ),
            "clipvalue": (
                model_params.optimizer_params.clipvalue
                if experiment.optimizer_finetuning
                else None
            ),
        }

        if experiment.include_budget:
            with tracker.budget.get_lock():
                row["budget"] = tracker.budget.value
                return row

        return row
