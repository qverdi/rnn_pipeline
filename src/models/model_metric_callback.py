from keras.callbacks import Callback
import time
from src.experiments.experiment_tracker import ExperimentTracker
from src.models.model_evaluation import ModelEvaluation
from src.models.model_metrics import ModelMetrics
from src.experiments.experiment_param import ExperimentParam
from src.utils.log_post import LogPost
import numpy as np
import src.utils.model_calculations as model_calculations

class MetricsCallback(Callback):
    """
    A custom Keras callback for evaluating and logging metrics at the end of each epoch.

    This callback evaluates the model's performance on test data, calculates metrics,
    and logs the results. It also checks and updates the best metric values if a
    better model performance is achieved during training.

    Attributes:
        data (train, val, test): The dataset used for evaluation.
        experiment (ExperimentParam): Parameters related to the experiment.
        logger (Logger): A logger instance for recording results.
        tracker (ExperimentTracker): A dictionary to track and compare the best metrics
                                achieved during training.
    """

    def __init__(
        self, data, experiment: ExperimentParam, logger: LogPost, tracker: ExperimentTracker, args:dict=None
    ):
        """
        Initializes the MetricsCallback.

        Args:
            data (tuple): The train, validation, and test dataset in the format (train_data, val_data, test_data).
            experiment (ExperimentParam): The experiment parameters, including training settings (e.g., patience, step_check).
            logger (Logger): Logger instance to log and append results.
            tracker (ExperimentTracker)
        """
        super().__init__()
        self.logger = logger  # Logger instance for recording metrics and results
        self.train_data, self.val_data, self.test_data = data  # Train, validation, and test data
        self.experiment = experiment  # Experiment parameters
        self.patience = self.experiment.patience  # Patience for early stopping
        self.step_check = self.experiment.step_check  # Step size for periodic evaluation
        self.start_time = time.time()  # Start time of the training process
        self.tracker = tracker  # Data for metric comparisons
        self.args = args
        self.history = {
            "train_loss": [],
            "val_loss": [],
            "train_mae": [],
            "val_mae": [],
            "train_mse": [],
            "val_mse": [],
            "train_smape": [],
            "val_smape": [],
            "train_rmse": [],
            "val_rmse": [],
            "train_aunl": [],
            "val_aunl": [],
        }

    def on_epoch_end(self, epoch, logs=None):
        """
        Executes at the end of each epoch. Evaluates the model, calculates metrics, and logs results.

        Args:
            epoch (int): The current epoch number (0-based).
            logs (dict, optional): Dictionary of logs containing epoch metrics such as loss and val_loss.

        Description:
            - Evaluates the model on train, validation, and test data.
            - Calculates AUNL (training and validation) and logs all relevant metrics.
            - Checks if it's the time to evaluate based on the step size or patience.
            - Updates best metrics if the current performance exceeds the previously stored best performance.
        """
        logs = logs or {}  # Initialize logs if not provided
        self.history["train_loss"].append(logs["loss"])  # Append the training loss
        self.history["val_loss"].append(logs["val_loss"])  # Append the validation loss

        aunl, aunl_val = model_calculations.calculate_aunl(
            self.history["train_loss"], self.history["val_loss"]
        )     
        train, val, test = self.get_evaluated_metrics(epoch)
   
        self.update_history((train, val), (aunl, aunl_val))
 
        should_evaluate = self.is_evaluation_epoch(epoch)
        self.log_metrics(epoch, logs, test, should_evaluate, (aunl, aunl_val))

        self.update_tracker(val, should_evaluate, epoch, aunl_val)
        
        self.update_best_weights(val)

        self.hpo_report(val, epoch)

    def get_evaluated_metrics(self, epoch):
        """
        Evaluates the model on the train, validation, and test datasets.

        Returns:
            tuple: Three dictionaries containing the evaluation metrics for train, validation, and test datasets.
        
        Description:
            - This function computes the evaluation metrics (e.g., MAE, MSE, etc.) on the train, validation, and test data.
        """
        # Evaluate the model on the train data
        model_evaluation = ModelEvaluation(self.train_data, self.experiment, self.model)
        train_metrics = model_evaluation.evaluate(epoch)

        # Evaluate the model on the validation data
        model_evaluation = ModelEvaluation(self.val_data, self.experiment, self.model)
        val_metrics = model_evaluation.evaluate(epoch)

        # Evaluate the model on the test data
        model_evaluation = ModelEvaluation(self.test_data, self.experiment, self.model)
        test_metrics = model_evaluation.evaluate(epoch)

        return train_metrics, val_metrics, test_metrics

    def is_evaluation_epoch(self, epoch):
        """
        Checks if the current epoch qualifies for evaluation based on patience and step size.

        Args:
            epoch (int): The current epoch number (0-based).
        
        Returns:
            bool: True if it's time to evaluate, otherwise False.

        Description:
            - Evaluates whether the model should be evaluated at the current epoch based on the patience and step_check parameters.
        """
        patience_passed = epoch + 1 >= self.patience
        is_step = (
            (epoch + 1 - self.patience) % self.step_check == 0
            if patience_passed
            else False
        )
        is_last_epoch = (
            epoch + 1 == self.experiment.num_epochs
        )  # Check if it's the last epoch
        return is_step or is_last_epoch  # Evaluate if it's a step or the last epoch

    def log_metrics(self, epoch, logs, test_metrics, should_evaluate, aunl_data):
        """
        Logs the metrics for the current epoch, including loss, AUNL, and other performance metrics.

        Args:
            epoch (int): The current epoch number (0-based).
            logs (dict): Dictionary containing the metrics for the current epoch.
            test_metrics (dict): Metrics calculated from the test dataset.
            should_evaluate (bool): Whether to evaluate based on the step size or last epoch condition.
            aunl_data (tuple): A tuple containing training and validation AUNL values.

        Description:
            - Logs the training time, loss, validation loss, and AUNL values.
            - Prints metrics for the current epoch.
            - Appends results to the logger for further analysis.
        """
        set = {}
        for key in test_metrics:
            set[key] = np.round(test_metrics[key], 4)

        aunl, aunl_val = aunl_data
        training_time = time.time() - self.start_time

        loss = np.inf if np.isnan(logs['loss']) else logs['loss']
        val_loss = np.inf if np.isnan(logs['val_loss']) else logs['val_loss']
        
        set["aunl"] = np.round(aunl, 4)
        set["aunl_val"] = np.round(aunl_val, 4)
        set["loss"] = np.round(loss, 4)
        set["val_loss"] = np.round(val_loss, 4)
        set["training_time"] = np.round(training_time, 4)
        set['epoch'] = epoch

        results = ModelMetrics()
        results.update(set)

        # Print metrics for the current epoch
        print(
            f"Epoch {epoch+1}: MAE={set['mae']}, MSE={set['mse']}, RMSE={set['rmse']}, SMAPE={set['smape']}, AUNL={aunl}, AUNL_VAL={aunl_val}"
        )

        if not self.experiment.include_budget:
            self.logger.append_results(should_evaluate, results, None)
        else:
            self.decrease_budget_and_append_logs(should_evaluate, results)


    def update_history(self, data, aunl_data):
        """
        Updates the history of metrics for both training and validation datasets, including AUNL values.

        Args:
            data (tuple): Tuple containing train and validation metrics.
            aunl_data (tuple): A tuple containing training and validation AUNL values.

        Description:
            - Appends the metrics for each epoch into the history dictionary for later analysis.
        """
        train_metrics, val_metrics = data
        aunl, aunl_val = aunl_data
        for key in train_metrics.keys():
            self.history[f"train_{key}"].append(np.round(train_metrics[key], 4))
            self.history[f"val_{key}"].append(np.round(val_metrics[key], 4))

        self.history["train_aunl"].append(np.round(aunl, 4))
        self.history["val_aunl"].append(np.round(aunl_val, 4))

    
    def update_tracker(self, val_metrics, should_evaluate, epoch, aunl_val):
        if self.experiment.early_stopping == "aunl":
            self.set_best_aunl(val_metrics, should_evaluate, epoch, aunl_val)
        elif self.experiment.early_stopping == "global_fuzzy_aunl":
            self.set_best_fuzzy_aunl(should_evaluate)


    def set_best_aunl(self, val_metrics, should_evaluate, epoch, aunl_val):
        """
        Updates the best model metrics if the current epoch improves performance.

        Args:
            val_metrics (dict): The validation metrics of the current epoch.
            should_evaluate (bool): Whether the evaluation condition is met.
            epoch (int): The current epoch number (0-based).
            aunl_val (float): The AUNL value for the current validation set.

        Description:
            - If the current validation performance surpasses the best stored performance, it updates the best metrics.
        """
        with self.tracker.best_value.get_lock():
            if (
                val_metrics[self.tracker.comparison_metric]
                < self.tracker.best_value.value
                and should_evaluate
            ):
                print(f"EPOCH: {epoch}/{self.experiment.num_epochs - 1}")
                print(
                    f"Best {self.tracker.comparison_metric}: {self.tracker.best_value.value}"
                )
                print(
                    f"Current best {self.tracker.comparison_metric}: {val_metrics[self.tracker.comparison_metric]}"
                )
                print(f"Current best AUNL: {aunl_val}")

                self.tracker.best_value.value = val_metrics[
                    self.tracker.comparison_metric
                ]
                with self.tracker.best_aunl.get_lock():
                    print(f"Best AUNL: {self.tracker.best_aunl.value}")
                    self.tracker.best_aunl.value = aunl_val


    def set_best_fuzzy_aunl(self, should_evaluate, window_size=3):
        if should_evaluate:
            aunl, aunl_val = model_calculations.calculate_aunl(self.history["train_loss"], self.history["val_loss"])
            
            # Calculate the current difference (generalization error)
            generalization_error = abs(aunl - aunl_val)

            if not hasattr(self, "generalization_errors"):
                self.generalization_errors = []

            # Append the current generalization error to the list
            self.generalization_errors.append(generalization_error)

                    # Ensure we have enough epochs to compare (at least `window_size` epochs)
            if len(self.generalization_errors) > window_size:
                # Calculate the average generalization error over the last `window_size` epochs
                current_window = self.generalization_errors[-window_size:]
                avg_current_window = np.mean(current_window)

                with self.tracker.best_ge.get_lock():
                    if avg_current_window < self.tracker.best_ge.value:
                        self.tracker.best_ge.value = avg_current_window

    
    def update_best_weights(self, val_metrics):
        if self.experiment.weight_sharing:
            with self.tracker.best_value.get_lock():
                if (val_metrics[self.tracker.comparison_metric]
                    < self.tracker.best_value.value):
                    with self.tracker.weights_lock:
                        self.tracker.best_weights = self.model.get_weights()
                    print(f"New best weights. Best metric value {self.tracker.best_value.value}")


    def hpo_report(self, val_metrics, epoch):
        aunl, aunl_val = model_calculations.calculate_aunl(self.history["train_loss"], self.history["val_loss"])
        new_metric = val_metrics[self.tracker.comparison_metric]
        
        if self.experiment.early_stopping == "aunl":
            new_metric = aunl_val

        if self.experiment.early_stopping == "global_fuzzy_aunl" or self.experiment.early_stopping == "naive_fuzzy_aunl":
            new_metric = abs(aunl - aunl_val)

        in_candidates = False

        if 'trial' in self.args:
             self.args['trial'].report(new_metric, epoch)
        elif 'model_id' in self.args:
            for i, candidate in enumerate(self.tracker.candidates):
                if candidate["model_id"] == self.args["model_id"]:
                    # Replace the dictionary entry at index i
                    self.tracker.candidates[i] = {
                        "model_id": candidate["model_id"],
                        "metric": new_metric,
                        "epoch": epoch
                    }
                    in_candidates = True
                    break

            if not in_candidates:
                print(f"[WARNING] Model ID {self.args['model_id']} not found in candidates.")

    
    def decrease_budget_and_append_logs(self, should_evaluate, results):
        with self.tracker.budget.get_lock():
            if self.tracker.budget.value >= 1:
                self.tracker.budget.value -= 1
                if self.tracker.budget.value >= 0:
                    self.logger.append_results(should_evaluate, results, self.tracker.budget.value)