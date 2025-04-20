from src.experiments.experiment_tracker import ExperimentTracker
from src.models.model_params import ModelParams
from src.experiments.experiment_param import ExperimentParam
from src.data.data_loader import DataLoader
from src.data.data_processing import DataProcessing
from src.data.data_container import DataContainer
from src.models.model import Model
from src.models.model_metric_callback import MetricsCallback
from src.models.model_stopping_callback import StoppingCallback
from src.utils.log_post import LogPost
from src.utils.value_mapper import ValueMapper
import tensorflow as tf
import numpy as np

tf.random.set_seed(123)


class ModelTraining:
    """
    A class to handle the training process for a machine learning model, including data loading,
    model compilation, and training.

    Attributes:
        experiment (ExperimentParam): Experiment configuration parameters.
        model_params (ModelParams): Model hyperparameters and configurations.
        logger (Logger): Logger to log the training process and metrics.
        data_container (DataContainer): Container holding processed training, validation, and test data.
        model: Compiled machine learning model ready for training.
    """

    def __init__(
        self,
        experiment: ExperimentParam,
        tracker: ExperimentTracker,
        model_params: ModelParams,
        args: dict = None,
    ):
        """
        Initializes the ModelTraining object with experiment and model parameters,
        loads data, and compiles the model.

        Args:
            experiment (ExperimentParam): Experiment configuration parameters.
            model_params (ModelParams): Model hyperparameters and configurations.
        """
        self.experiment = experiment
        self.tracker = tracker
        self.args = args

        # Load and process data
        self.data_container = self.load_data()
        print("Data loaded successfully.")

        # Compile model and load weights
        self.model = self.load_model(model_params)
        print("Model loaded successfully.")

    def load_model(self, model_params):
        self.model_params = model_params
        self.model_params.input_shape = self.data_container.get_input_shape()

        # Initialize the model
        model = Model(self.model_params)

        if self.experiment.sh:
            self.tracker.candidates.append(
                {"model_id": model.id, "metric": np.inf, "epoch": 0}
            )
            self.args["model_id"] = model.id

        # Initialize the logger with experiment and model details
        self.logger = LogPost(
            ValueMapper.get_log_dict(
                self.experiment, self.tracker, model_params, model.id
            ),
            self.tracker.log_queue,
        )

        # Build and compile the model
        model.build_model()
        print("Model compiled successfully.")

        with self.tracker.weights_lock:
            # Load best weights
            if self.experiment.weight_sharing:
                model.set_weights(self.tracker.best_weights)

        return model.get_model()

    def load_data(self):
        """
        Loads and processes the data required for training, validation, and testing.

        Returns:
            DataContainer: A container holding processed training, validation, and test data.

        Description:
            - This method loads raw data using `DataLoader`, processes it using `DataProcessing`,
              and then packages the processed data into a `DataContainer` object.
        """
        # Load raw data
        data_loader = DataLoader(self.experiment)
        data = data_loader.get_data()

        # Process the data
        data_processing = DataProcessing(data, self.experiment)
        train_data, val_data, test_data = data_processing.process_data()

        # Create a data container for the processed data
        data_container = DataContainer(train_data, val_data, test_data)

        return data_container

    def train_model(self):
        """
        Trains the model using the training and validation data, and logs the metrics.

        Args:
            tracker (ExperimentTracker): Data for metric comparisons, such as the best model's AUNL.

        Returns:
            dict: A dictionary containing the final values of training metrics after the last epoch.

        Description:
            - This function performs the training loop for the model.
            - It calls `initialize_callbacks` to set up the necessary callbacks and logs the metrics.
            - After training, it retrieves the final objective score.
        """
        # Retrieve training, validation, and test data
        X_train, y_train = self.data_container.get_train_data()
        X_val, y_val = self.data_container.get_val_data()
        X_test, y_test = self.data_container.get_test_data()

        data = ((X_train, y_train), (X_val, y_val), (X_test, y_test))

        # Set batch size
        batch_size = self.model_params.batch_size

        # Initialize metrics callback
        callbacks = self.initialize_callbacks(data)

        print("Training started.")

        # Train the model
        history = self.model.fit(
            X_train,
            y_train,
            epochs=self.experiment.num_epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            verbose=1,
            callbacks=callbacks,
        )
        print("Training finished.")

        # Retrieve the objective function value
        objective = self.get_fitness_objective(callbacks)
        return objective

    def initialize_callbacks(self, data):
        """
        Initialize and return the list of callbacks.

        Args:
            data (tuple): The training, validation, and test data in the form ((X_train, y_train), (X_val, y_val), (X_test, y_test)).

        Returns:
            list: List of callbacks to be used during training.

        Description:
            - This function initializes necessary callbacks for model evaluation, stopping conditions,
              and budget management. It prepares the callbacks based on the experiment's configuration.
        """
        callbacks = []

        # Evaluation callbacks
        callbacks.append(
            MetricsCallback(data, self.experiment, self.logger, self.tracker, self.args)
        )

        if (
            self.experiment.early_stopping != "none"
            or (
                self.experiment.include_budget
                and self.experiment.budget_strategy == "fixed"
            )
            or self.experiment.sh
        ):
            callbacks.append(StoppingCallback(self.experiment, self.tracker, self.args))

        return callbacks

    def get_fitness_objective(self, callbacks):
        """
        Calculates the fitness objective based on experiment settings

        Args:
            callbacks (list): List of callbacks used during training.

        Returns:
            float: The fitness objective value (e.g., a weighted sum or singular metric).

        Description:
            - This function computes the fitness objective based on the configured type.
            - It handles both singular and weighted objectives, which can be a combination of metrics like loss, AUNL, etc.
        """
        history = callbacks[0].history

        objective = self.experiment.objective.split("_")
        last_values = {metric: values[-1] for metric, values in history.items()}

        if objective[0] == "singular":
            metric = objective[1]
            train = last_values[f"train_{metric}"]
            val = last_values[f"val_{metric}"]
            return np.round((train + val) / 2, 4)

        elif objective[0] == "weighted":
            metric = objective[1]
            concept = objective[2]

            return self.get_weighted_sum(history, metric, concept)

        return last_values["val_loss"]

    def get_weighted_sum(self, history, metric, concept):
        """
        Computes a weighted sum based on the specified concept (e.g., stability, complexity, generalization).

        Args:
            history (dict): The history of metrics during training.
            metric (str): The metric name (e.g., "loss", "mae").
            concept (str): The concept for the weighted sum calculation (e.g., "stability", "complexity", "generalization").

        Returns:
            float: The computed weighted sum based on the specified concept.

        Description:
            - This function computes different weighted sums for stability, complexity, or generalization
              based on the model's training and validation metrics.
        """
        w1 = self.experiment.objective_weights[0]
        w2 = self.experiment.objective_weights[1]

        if concept == "stability":
            history_metric = history[f"val_{metric}"]
            train_metric = history[f"train_{metric}"][-1]
            val_metric = history[f"val_{metric}"][-1]
            return w1 / 2 * (train_metric + val_metric) + w2 * np.std(history_metric)

        if concept == "complexity":
            train_metric = history[f"train_{metric}"][-1]
            val_metric = history[f"val_{metric}"][-1]

            num_params = sum(
                [
                    tf.reduce_prod(var.shape).numpy()
                    for var in self.model.trainable_variables
                ]
            )
            return w1 / 2 * (train_metric + val_metric) + w2 * num_params

        if concept == "generalization":
            train_metric = history[f"train_{metric}"][-1]
            val_metric = history[f"val_{metric}"][-1]

            return w1 / 2 * (train_metric + val_metric) + w2 * abs(
                train_metric - val_metric
            )
