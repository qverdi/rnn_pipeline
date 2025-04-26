from src.models.model_optimizer_params import ModelOptimizerParams
from src.models.model_params import ModelParams
import numpy as np
import src.utils.json_utils as json_utils
from src.config.file_constants import MODEL_DESIGN
import optuna
from src.experiments.experiment_param import ExperimentParam
from src.optimization.hpo_optimizer_set import HPOOptimizerSet


class SearchOptimizerSet(HPOOptimizerSet):
    """
    Manages the generation of parameter sets for model compilation for random/bayesian optimization during model training.

    This class facilitates the creation of parameter sets that are dynamically adapted based on whether
    the experiment has known model architecture or not, and the configuration of the search space. It also handles
    frozen and unfrozen model layers.

    Attributes:
        experiment (ExperimentParam): Configuration for the current experiment, including model details and hyperparameters.
        value_mapper (ValueMapper): A utility to map and denormalize parameter values.
    """

    def __init__(self, search_space: dict, experiment: ExperimentParam):
        super().__init__()
        """
        Initialize the optimizer with the search space and experiment parameters.

        Args:
            search_space (dict): The search space containing possible hyperparameter values.
            experiment (ExperimentParam): An instance of the ExperimentParam class containing experiment details.
        """
        self.search_space = search_space
        self.experiment = experiment

        if self.experiment.optimizer_finetuning:
            search_space.update(ModelOptimizerParams.get_params())

    def create_model_params(self, trial: optuna.Trial) -> ModelParams:
        """
        Creates the model parameters based on the trial's suggestions.

        Args:
            trial (optuna.Trial): The trial object representing a single optimization iteration.

        Returns:
            ModelParams: The model parameters based on the current trial's suggested values.
        """
        if self.experiment.has_model:
            return ModelParams(self.generate_specific_set(trial))
        else:
            return ModelParams(self.generate_generic_set(trial))

    def generate_generic_set(self, trial: optuna.Trial) -> dict:
        """
        Generates a set of hyperparameters for a generic model architecture.

        Args:
            trial (optuna.Trial): The trial object for generating hyperparameter suggestions.

        Returns:
            dict: A dictionary containing the hyperparameters suggested by the trial.
        """
        descrete = self.experiment.is_descrete
        numerical_params = {
            "num_layers": int,
            "neurons": int,
            "batch_size": int,
            "dropout_rate": float,
            "learning_rate": float,
        }
        categorical_params = ["activation", "layers", "optimizer"]

        set = {}
        for param, dtype in numerical_params.items():
            if descrete:
                set[param] = trial.suggest_categorical(param, self.search_space[param])
            else:
                suggest_func = trial.suggest_int if dtype == int else trial.suggest_float
                set[param] = round(
                    suggest_func(param, min(self.search_space[param]), max(self.search_space[param])), 
                    4 if dtype == float else 0
                )

        for param in categorical_params:
            set[param] = trial.suggest_categorical(param, self.search_space[param])

        # Expand layer-specific parameters
        for param in ["neurons", "layers", "dropout_rate", "activation"]:
            set[param] = np.full(set["num_layers"], set[param]).tolist()

        set["experiment_id"] = self.experiment.id
        set["loss"] = self.experiment.loss_function
        set["horizon"] = self.experiment.horizon

        if self.experiment.optimizer_finetuning:
            set.update(self.generate_model_optimizer_set(trial))

        return set
    
    def generate_specific_set(self, trial: optuna.Trial) -> dict:
        """
        Generates a set of hyperparameters, considering model-specific design and frozen parameters.

        Args:
            trial (optuna.Trial): The trial object for generating hyperparameter suggestions.

        Returns:
            dict: A dictionary containing the model-specific hyperparameters.
        """
        model_design = json_utils.read_json_file(MODEL_DESIGN)

        set = self.initialize_base_set(trial)
        set.update(model_design)

        if "freeze" in model_design:
            freeze = model_design["freeze"]
            keys = self.get_layer_params()

            for key in keys:
                set[key] = self.combine_frozen_and_unfrozen(
                    trial, key, freeze, model_design
                )

        set["num_layers"] = len(set["layers"])
        
        if self.experiment.optimizer_finetuning:
            set.update(self.generate_model_optimizer_set(trial))
        
        return set

    def initialize_base_set(self, trial: optuna.Trial) -> dict:
        """
        Initializes the basic set of hyperparameters for the model.

        Args:
            trial (optuna.Trial): The trial object for generating hyperparameter suggestions.

        Returns:
            dict: A dictionary with basic hyperparameters like batch_size, learning_rate, and optimizer.
        """
        descrete = self.experiment.is_descrete
        param_types = {
            "batch_size": int,
            "learning_rate": float,
        }

        base_set = {}
        for param, dtype in param_types.items():
            if descrete:
                base_set[param] = trial.suggest_categorical(param, self.search_space[param])
            else:
                suggest_func = trial.suggest_int if dtype == int else trial.suggest_float
                base_set[param] = round(
                    suggest_func(param, min(self.search_space[param]), max(self.search_space[param])), 
                    4 if dtype == float else 0
                )

        base_set["optimizer"] = trial.suggest_categorical("optimizer", self.search_space["optimizer"])
        base_set.update({
            "experiment_id": self.experiment.id,
            "loss": self.experiment.loss_function,
            "horizon": self.experiment.horizon
        })

        return base_set


    def combine_frozen_and_unfrozen(
        self, trial: optuna.Trial, key: str, freeze: list, model_design: dict
    ) -> list:
        """
        Combines frozen and unfrozen values for a given parameters set for model compilation.

        Args:
            trial (optuna.Trial): The trial object for generating hyperparameter suggestions.
            key (str): The key of the hyperparameter to combine.
            freeze (list): A list indicating which layers are frozen.
            model_design (dict): The model design containing the frozen and unfrozen parameters.

        Returns:
            list: A list of combined frozen and unfrozen values for the given hyperparameter.
        """
        unfrozen_values = self.get_unfrozen_values(trial, key, freeze)
        combined_values = []

        for idx, is_frozen in enumerate(freeze):
            if is_frozen == 1:
                combined_values.append(model_design[key][idx])  # Use frozen value
            else:
                combined_values.append(unfrozen_values.pop(0))  # Use suggested value
        return combined_values

    def get_unfrozen_values(self, trial: optuna.Trial, key: str, freeze: list) -> list:
        """
        Extracts the unfrozen values for a parameter set based on the freeze configuration.

        Args:
            trial (optuna.Trial): The trial object for generating hyperparameter suggestions.
            key (str): The key of the hyperparameter to extract unfrozen values for.
            freeze (list): A list indicating which values are frozen.

        Returns:
            list: A list of unfrozen suggested values for the hyperparameter.
        """
        unfrozen_values = []
        for idx, is_frozen in enumerate(freeze):
            if is_frozen == 0:
                key_idx = f"{key}_{idx}"
                suggested_value = self.suggest_value(trial, key, key_idx)
                unfrozen_values.append(suggested_value)
        return unfrozen_values

    
    def suggest_value(self, trial: optuna.Trial, key, key_idx):
        """
        Suggest value based on key and `is_descrete` property.

        Args:
            trial (optuna.Trial): The trial object for generating hyperparameter suggestions.
            key (str): The key of the hyperparameter to extract unfrozen values for.
            key_idx (str): The key of the hyperparameter and layer index.

        Returns:
            any: Suggested value.
        """
        discrete = self.experiment.is_descrete

        param_types = {
            "neurons": int,
            "dropout_rate": float,
        }

        if key in param_types:
            suggest_func = trial.suggest_int if param_types[key] == int else trial.suggest_float
            value = (
                trial.suggest_categorical(key_idx, self.search_space[key])
                if discrete
                else suggest_func(key_idx, min(self.search_space[key]), max(self.search_space[key]))
            )
            return round(value, 4) if param_types[key] == float else value

        if key in ["activation", "layers"]:
            return trial.suggest_categorical(key_idx, self.search_space[key])

        raise ValueError(f"Unsupported key: {key}")
    
    def generate_model_optimizer_set(self, trial: optuna.Trial,):
        """
        Generates a set of hyperparameters for a generic model architecture.

        Args:
            trial (optuna.Trial): The trial object for generating hyperparameter suggestions.

        Returns:
            dict: A dictionary containing the hyperparameters suggested by the trial.
        """
        descrete = self.experiment.is_descrete
        numerical_params = {
            "beta_1": float,
            "beta_2": float,
            "rho": float,
            "weight_decay": float,
            "clipnorm": float,
            "clipvalue": float
        }
        categorical_params = ["centered", "nesterov", "amsgrad", "clipping"]

        set = {}
        for param, dtype in numerical_params.items():
            if descrete:
                set[param] = trial.suggest_categorical(param, self.search_space[param])
            else:
                suggest_func = trial.suggest_int if dtype == int else trial.suggest_float
                set[param] = round(
                    suggest_func(param, min(self.search_space[param]), max(self.search_space[param])), 
                    4 if dtype == float else 0
                )

        for param in categorical_params:
            set[param] = trial.suggest_categorical(param, self.search_space[param])

        return set
