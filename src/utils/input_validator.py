from src.config.file_constants import (
    MODEL_OPTIMIZER_PARAMS,
    SEARCH_SPACE_FILE,
    EXPERIMENT_PARAMS,
    GA_SEARCH_OPTIMIZER_PARAMS,
    DE_SEARCH_OPTIMIZER_PARAMS,
    SEARCH_OPTIMIZER_PARAMS,
    OUTPUT_DIR,
    MODEL_DESIGN,
)
import src.utils.json_utils as json_utils
import os


class InputValidator:
    """
    Validates input JSON files.
    """
    def __init__(self):
        self.__load_files()
        self.validate()

    def __load_files(self):
        self.experiment_params = json_utils.read_json_file(EXPERIMENT_PARAMS)
        self.model_params = json_utils.read_json_file(SEARCH_SPACE_FILE)
        self.model_optimizer_params = json_utils.read_json_file(MODEL_OPTIMIZER_PARAMS)
        self.model_design = json_utils.read_json_file(MODEL_DESIGN)
        self.search_params = json_utils.read_json_file(SEARCH_OPTIMIZER_PARAMS)
        self.ga_params = json_utils.read_json_file(GA_SEARCH_OPTIMIZER_PARAMS)
        self.de_params = json_utils.read_json_file(DE_SEARCH_OPTIMIZER_PARAMS)

    def validate(self):
        self.validate_experiment()

        self.validate_search_algorithm()

        if self.experiment_params["has_model"]:
            self.validate_model_architecture()

        if self.experiment_params["include_budget"]:
            self.validate_budget()

        if self.experiment_params["optimizer_finetuning"]:
            self.validate_model_optimizer_params()

        self.validate_model_params()

    def validate_experiment(self):
        data = self.experiment_params

        required_keys = [
            "gpu",
            "loss",
            "num_epochs",
            "optimizer",
            "patience",
            "train_size",
            "step_check",
            "val_size",
            "window_size",
            "has_model",
            "is_descrete",
            "early_stopping",
            "include_budget",
            "reference_metric",
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in experiment_params.")

        # Check data types
        if not isinstance(data.get("gpu"), int):
            raise ValueError("gpu must be an integer.")
        if not isinstance(data.get("loss"), str):
            raise ValueError("loss must be a string.")
        if not isinstance(data.get("num_epochs"), int):
            raise ValueError("num_epochs must be an integer.")
        if not isinstance(data.get("optimizer"), str):
            raise ValueError("optimizer must be a string.")
        if not isinstance(data.get("patience"), int):
            raise ValueError("patience must be an integer.")
        if not isinstance(data.get("train_size"), float):
            raise ValueError("train_size must be a float.")
        if not isinstance(data.get("step_check"), int):
            raise ValueError("step_check must be an integer.")
        if not isinstance(data.get("val_size"), float):
            raise ValueError("val_size must be a float.")
        if not isinstance(data.get("window_size"), int):
            raise ValueError("window_size must be an integer.")
        if not isinstance(data.get("has_model"), bool):
            raise ValueError("has_model must be a bool.")
        if not isinstance(data.get("is_descrete"), bool):
            raise ValueError("is_descrete must be a bool.")
        if not isinstance(data.get("early_stopping"), str):
            raise ValueError("early_stopping must be a str.")
        if not isinstance(data.get("include_budget"), bool):
            raise ValueError("include_budget must be a bool.")

        # Validate specific values
        if data["optimizer"] not in ["search"]:
            raise ValueError("optimizer must be 'search'")
        if data["loss"] not in ["mae", "mse"]:
            raise ValueError("loss must be 'mse' or 'mae'")
        if not (0 < data["train_size"] < 1):
            raise ValueError("train_size must be between 0 and 1.")
        if not (0 < data["val_size"] < 1):
            raise ValueError("val_size must be between 0 and 1.")
        if data["train_size"] + data["val_size"] > 0.9:
            raise ValueError("train_size and val_size combined cannot exceed 0.9.")
        if data["patience"] > data["num_epochs"]:
            raise ValueError("patience cannot be greater than num_epochs.")
        if data["step_check"] > data["num_epochs"]:
            raise ValueError("step_check cannot be greater than num_epochs.")
        if data["patience"] + data["step_check"] >= data["num_epochs"]:
            raise ValueError("patience + step_check must be less than num_epochs.")
        if not isinstance(data.get("reference_metric"), str):
            raise ValueError("reference_metric must be an str.")
        
        early_stopping = [
            "none",
            "default",
            "aunl",
            "naive_fuzzy_aunl",
            "global_fuzzy_aunl",
            "predictive"
        ]

        if data["early_stopping"] not in early_stopping:
            raise ValueError(f"early_stopping not valid. Available early_stopping {early_stopping}")
        
        if data["reference_metric"] not in ["mae", "mse"]:
            raise ValueError("reference_metric must be 'mae' or 'mse'.")


    def validate_budget(self):
        data = self.experiment_params

        required_keys = [
            "budget_value",
            "budget_strategy",
            "budget_tolerance",
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in experiment_params.")

        if not isinstance(data.get("budget_value"), int):
            raise ValueError("budget_value must be an int.")
        if not isinstance(data.get("budget_strategy"), str):
            raise ValueError("budget_strategy must be an str.")
        if not isinstance(data.get("budget_tolerance"), int):
            raise ValueError("budget_tolerance must be an int.")

        if data["budget_strategy"] not in ["fixed", "greedy"]:
            raise ValueError("budget_strategy must be 'fixed' or 'greedy'.")

        if self.experiment_params["early_stopping"] == "aunl":
            if self.experiment_params["optimizer"] == "search":
                if (
                    data["budget_value"]
                    > self.experiment_params["patience"] * self.search_params["trials"]
                ):
                    raise ValueError("budget must be smaller than total estimated epochs.")

        if data['budget_tolerance'] > data['budget_value']:
            raise ValueError("tolerance must be smaller than budget.")

    
    def validate_model_optimizer_params(self):
        data = self.model_optimizer_params

        required_keys = [
            "beta_1", "beta_2", "rho", "momentum", "centered", "weight_decay",
            "nesterov", "amsgrad", "clipnorm", "clipvalue", "clipping"
        ]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in model optimizer parameters.")

        # Helper function to check if a value is a list of floats
        def is_list_of_floats(value):
            return isinstance(value, list) and all(isinstance(i, float) for i in value)

        # Validate beta_1
        if not is_list_of_floats(data.get("beta_1")):
            raise ValueError("beta_1 must be a list of floats.")
        if not all(0 < val < 1 for val in data["beta_1"]):
            raise ValueError("Each value in beta_1 must be between 0 and 1 (exclusive).")

        # Validate beta_2
        if not is_list_of_floats(data.get("beta_2")):
            raise ValueError("beta_2 must be a list of floats.")
        if not all(0 < val < 1 for val in data["beta_2"]):
            raise ValueError("Each value in beta_2 must be between 0 and 1 (exclusive).")

        # Validate rho
        if not is_list_of_floats(data.get("rho")):
            raise ValueError("rho must be a list of floats.")
        if not all(0 < val < 1 for val in data["rho"]):
            raise ValueError("Each value in rho must be between 0 and 1 (exclusive).")

        # Validate momentum
        if not is_list_of_floats(data.get("momentum")):
            raise ValueError("momentum must be a list of floats.")
        if not all(0 <= val <= 1 for val in data["momentum"]):
            raise ValueError("Each value in momentum must be between 0 and 1 (inclusive).")

        # Validate centered
        if not isinstance(data.get("centered"), list) or not all(isinstance(i, bool) for i in data["centered"]):
            raise ValueError("centered must be a list of boolean values.")

        # Validate weight_decay
        if not is_list_of_floats(data.get("weight_decay")):
            raise ValueError("weight_decay must be a list of floats.")
        if not all(0 <= val <= 1e-3 for val in data["weight_decay"]):
            raise ValueError("Each value in weight_decay must be between 0 and 1e-3 (inclusive).")

        # Validate nesterov
        if not isinstance(data.get("nesterov"), list) or not all(isinstance(i, bool) for i in data["nesterov"]):
            raise ValueError("nesterov must be a list of boolean values.")

        # Validate amsgrad
        if not isinstance(data.get("amsgrad"), list) or not all(isinstance(i, bool) for i in data["amsgrad"]):
            raise ValueError("amsgrad must be a list of boolean values.")

        # Validate clipnorm
        if not is_list_of_floats(data.get("clipnorm")):
            raise ValueError("clipnorm must be a list of floats.")
        if not all(0 <= val <= 10.0 for val in data["clipnorm"]):
            raise ValueError("Each value in clipnorm must be between 0 and 10 (inclusive).")

        # Validate clipvalue
        if not is_list_of_floats(data.get("clipvalue")):
            raise ValueError("clipvalue must be a list of floats.")
        if not all(0 <= val <= 1.0 for val in data["clipvalue"]):
            raise ValueError("Each value in clipvalue must be between 0 and 1 (inclusive).")

        # Validate clipping
        if not isinstance(data.get("clipping"), list) or not all(
            val in ["clipnorm", "clipvalue"] for val in data["clipping"]
        ):
            raise ValueError("clipping must be a list containing 'clipnorm' or 'clipvalue'.")


    
    def validate_search_algorithm(self):
        data = self.search_params

        required_keys = ["sampler", "trials"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in search algorithm.")

        if not isinstance(data.get("sampler"), str):
            raise ValueError("sampler must be a string.")
        if not isinstance(data.get("trials"), int):
            raise ValueError("trials must be an integer.")

        if data["sampler"] not in ["random", "bayesian"]:
            raise ValueError("sampler must be 'random' or 'bayesian'.")

        if data["trials"] <= 0:
            raise ValueError("trials must be greater than 0.")

    def validate_model_architecture(self):
        data = self.model_design

        # Validate presence of required keys
        required_keys = ["layers", "neurons", "dropout_rate", "activation"]
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in model_architecture.")

        # Validate data types
        if not isinstance(data["layers"], list) or not all(
            isinstance(l, str) for l in data["layers"]
        ):
            raise ValueError("layers must be a list of strings.")

        if not isinstance(data["neurons"], list) or not all(
            isinstance(n, int) for n in data["neurons"]
        ):
            raise ValueError("neurons must be a list of integers.")

        if not isinstance(data["dropout_rate"], list) or not all(
            isinstance(d, float) for d in data["dropout_rate"]
        ):
            raise ValueError("dropout must be a list of floats.")

        if not isinstance(data["activation"], list) or not all(
            isinstance(a, str) for a in data["activation"]
        ):
            raise ValueError("activation must be a list of strings.")

        # Validate `layers`
        self.is_layer_name_valid(data["layers"])

        # Validate `activation`
        self.is_activation_valid(data["activation"])

        # Validate lengths
        lengths = [len(data[key]) for key in required_keys]
        if all(length > 1 for length in lengths):
            if len(set(lengths)) > 1:
                raise ValueError(
                    "All arrays in model_architecture must have the same length when their lengths are greater than 1."
                )

        # Validate `freeze` if present
        if "freeze" in data:
            if not isinstance(data["freeze"], list) or not all(
                isinstance(f, int) and f in [0, 1] for f in data["freeze"]
            ):
                raise ValueError("freeze must be a list of binary integers (0 or 1).")

            if len(data["freeze"]) != lengths[0]:
                raise ValueError(
                    "The length of the freeze array must match the length of the other arrays in model_architecture."
                )

    def is_activation_valid(self, activations):
        # Validate `activation`
        valid_activations = {
            "elu",
            "exponential",
            "gelu",
            "hard_sigmoid",
            "hard_silu",
            "hard_swish",
            "leaky_relu",
            "linear",
            "log_softmax",
            "mish",
            "relu",
            "relu6",
            "selu",
            "sigmoid",
            "silu",
            "softmax",
            "softplus",
            "softsign",
            "swish",
            "tanh",
        }
        if not all(act in valid_activations for act in activations):
            raise ValueError(
                f"activation can only contain the following values: {valid_activations}."
            )

    def is_layer_name_valid(self, layers):
        valid_layers = {"rnn", "lstm", "gru", "dense"}
        if not all(layer in valid_layers for layer in layers):
            raise ValueError(
                f"layers can only contain the following values: {valid_layers}."
            )

    def validate_model_params(self):
        data = self.model_params

        required_keys = [
            "activation",
            "batch_size",
            "dropout_rate",
            "learning_rate",
            "layers",
            "num_layers",
            "neurons",
            "optimizer",
        ]

        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key '{key}' in hyperparameters.")

        # Validate `activation`
        self.is_activation_valid(data["activation"])

        # Validate `batch_size`
        if not isinstance(data["batch_size"], list) or not all(
            isinstance(b, int) and b > 0 for b in data["batch_size"]
        ):
            raise ValueError("batch_size must be a list of positive integers.")

        # Validate `dropout_rate`
        if not isinstance(data["dropout_rate"], list) or not all(
            isinstance(d, float) and 0 <= d < 1 for d in data["dropout_rate"]
        ):
            raise ValueError("dropout_rate must be a list of floats between 0 and 1.")

        # Validate `learning_rate`
        if not isinstance(data["learning_rate"], list) or not all(
            isinstance(lr, float) and 0 < lr < 1 for lr in data["learning_rate"]
        ):
            raise ValueError(
                "learning_rate must be a list of floats greater than 0 and less than 1."
            )

        # Validate `layers`
        self.is_layer_name_valid(data["layers"])

        # Validate `num_layers`
        if not isinstance(data["num_layers"], list) or not all(
            isinstance(nl, int) and nl > 0 for nl in data["num_layers"]
        ):
            raise ValueError("num_layers must be a list of positive integers.")

        # Validate `neurons`
        if not isinstance(data["neurons"], list) or not all(
            isinstance(nn, int) and nn > 0 for nn in data["neurons"]
        ):
            raise ValueError("neurons must be a list of positive integers.")

        # Validate `optimizer`
        valid_optimizers = {"adam", "rmsprop", "sgd"}
        if not isinstance(data["optimizer"], list) or not all(
            isinstance(opt, str) and opt in valid_optimizers
            for opt in data["optimizer"]
        ):
            raise ValueError(
                f"optimizer must be a list of valid optimizers: {valid_optimizers}."
            )