from keras.callbacks import Callback
from src.experiments.experiment_param import ExperimentParam
from src.experiments.experiment_tracker import ExperimentTracker
import src.utils.model_calculations as model_calculations
import numpy as np
import pandas as pd
import optuna


class StoppingCallback(Callback):
    """
    A custom Keras callback to handle early stopping based on AUNL
    and fixed budget constraints during training.

    Attributes:
        experiment (ExperimentParam): Experiment parameters, including training settings.
        patience (int): Number of epochs to wait before checking for early stopping.
        loss (list): List to store training loss for each epoch.
        val_loss (list): List to store validation loss for each epoch.
        tracker (ExperimentTracker): The tracker with best values observed during training.
    """

    def __init__(
        self, experiment: ExperimentParam, tracker: ExperimentTracker, args=None
    ):
        """
        Initializes the StoppingCallback.

        Args:
            experiment (ExperimentParam): The experiment parameters, including budget and stopping criteria.
            tracker (ExperimentTracker): The tracker with best values observed during training.
        """
        super().__init__()
        self.experiment = experiment  # Experiment parameters
        self.patience = (
            self.experiment.patience
        )  # Number of epochs to wait before early stopping
        self.step_check = self.experiment.step_check
        self.loss = []  # List of training loss
        self.val_loss = []  # List of validation loss
        self.tracker = tracker  # Best AUNL value observed
        self.best = np.Inf
        self.min_delta = 0.001
        self.wait = 0
        self.args = args

    def on_epoch_end(self, epoch, logs=None):
        """
        Executes at the end of each epoch. Checks for early stopping conditions.

        Args:
            epoch (int): The current epoch number (0-based).
            logs (dict, optional): Dictionary of logs containing epoch metrics such as loss and val_loss.
        """
        logs = logs or {}  # Initialize logs if not provided
        self.loss.append(logs["loss"])  # Append the training loss
        self.val_loss.append(logs["val_loss"])  # Append the validation loss

        # Check if the budget strategy is fixed and terminate training if necessary
        if (
            self.experiment.include_budget
            and self.experiment.budget_strategy == "fixed"
        ):
            self.should_terminate(epoch)

        if self.experiment.early_stopping != "none":
            self.early_stopping(epoch, logs)

        if self.experiment.sh:
            self.sh_stopping(epoch)

    def should_terminate(self, epoch):
        """
        Terminates training if the budget is exhausted.

        Args:
            epoch (int): The current epoch number (0-based).
        """
        with self.tracker.budget.get_lock():
            if self.tracker.budget.value <= 0:
                self.stopped_epoch = epoch  # Store the epoch where training stopped
                self.model.stop_training = True  # Stop the training process

        
    def should_evaluate(self, epoch):
        # Determine if the patience period has passed and if the current epoch is a step for evaluation
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

    def early_stopping(self, epoch, logs):
        if self.experiment.early_stopping == "default":
            self.default_stopping(epoch, logs)
        elif self.experiment.early_stopping == "aunl":
            self.aunl_stopping(epoch)
        elif self.experiment.early_stopping == "naive_fuzzy_aunl":
            self.naive_fuzzy_aunl_stopping(epoch)
        elif self.experiment.early_stopping == "global_fuzzy_aunl":
            self.fuzzy_aunl_stopping(epoch)

    def default_stopping(self, epoch, logs):
        current = self.val_loss[-1]
        improvement = False

        if epoch + 1 <= self.patience:
            return

        if current < self.best - self.min_delta:
            self.best = current
            self.wait = 0
            improvement = True

        if not improvement:
            self.wait += 1

        if self.wait >= self.step_check:
            self.stopped_epoch = epoch  # Store the epoch where training stopped
            self.model.stop_training = True  # Stop the training process
            print(
                f"DEFAULT: Early stopping at epoch {epoch + 1}, no improvement in val loss."
            )

    def aunl_stopping(self, epoch):
        if not self.should_evaluate(epoch):
            return

        # Calculate the current AUNL values based on loss
        _, aunl_val = model_calculations.calculate_aunl(self.loss, self.val_loss)

        # If the validation AUNL worsens compared to the best observed value, stop training
        with self.tracker.best_aunl.get_lock():
            if aunl_val > self.tracker.best_aunl.value:
                self.stopped_epoch = epoch  # Store the epoch where training stopped
                self.model.stop_training = True  # Stop the training process
                print(f"AUNL: Early stopping at epoch {epoch + 1}, no improvement in AUNL.")

    def fuzzy_aunl_stopping(self, epoch, window_size=3):
        """
        Monitors the difference between aunl and aunl_val and stops training if the
        generalization error (difference between aunl and aunl_val) starts increasing
        over the last few epochs (rolling window).

        Args:
            epoch (int): The current epoch number.
            window_size (int): Number of last epochs to consider for the moving average of generalization error.
        """
        if not self.should_evaluate(epoch):
            return

        # Calculate the current AUNL values based on loss
        aunl, aunl_val = model_calculations.calculate_aunl(self.loss, self.val_loss)

        # Calculate the current difference (generalization error)
        generalization_error = abs(aunl - aunl_val)

        # Initialize the list to store generalization errors across epochs
        if not hasattr(self, "generalization_errors"):
            self.generalization_errors = []

        # Append the current generalization error to the list
        self.generalization_errors.append(generalization_error)

        # Ensure we have enough epochs to compare (at least `window_size` epochs)
        if len(self.generalization_errors) > window_size:
            # Calculate the average generalization error over the last `window_size` epochs
            current_window = self.generalization_errors[-window_size:]

            # Calculate the average error for the current and previous windows
            avg_current_window = np.mean(current_window)

            with self.tracker.best_ge.get_lock():
                if avg_current_window > self.tracker.best_ge.value:
                    self.stopped_epoch = epoch  # Store the epoch where training stopped
                    self.model.stop_training = True  # Stop the training process
                    print(
                        f"Fuzzy AUNL: Early stopping at epoch {epoch + 1}, no improvement in AUNL."
                    )

        print(
            f"Epoch {epoch+1}: AUNL={aunl}, AUNL_VAL={aunl_val}, Generalization error={generalization_error}"
        )

    def naive_fuzzy_aunl_stopping(self, epoch):
        """
        Monitors the difference between aunl and aunl_val and stops training if the
        generalization error (difference between aunl and aunl_val) starts increasing
        over the last few epochs (rolling window).

        Args:
            epoch (int): The current epoch number.
            window_size (int): Number of last epochs to consider for the moving average of generalization error.
        """
        if not self.should_evaluate(epoch):
            return

        # Calculate the current AUNL values based on loss
        aunl, aunl_val = model_calculations.calculate_aunl(self.loss, self.val_loss)

        # Calculate the current difference (generalization error)
        generalization_error = abs(aunl - aunl_val)

        # Initialize the list to store generalization errors across epochs
        if not hasattr(self, "generalization_errors"):
            self.generalization_errors = []

        # Append the current generalization error to the list
        self.generalization_errors.append(generalization_error)

        # Ensure we have enough epochs to compare (at least `window_size` epochs)
        if len(self.generalization_errors) > 1:
            # Calculate the average generalization error over the last `window_size` epochs
            current_window = self.generalization_errors[-1]
            previous_window = self.generalization_errors[-2]

            # Calculate the average error for the current and previous windows
            avg_current_window = np.mean(current_window)
            avg_previous_window = np.mean(previous_window)

            # If the average generalization error in the current window is larger than in the previous window, stop training
            if avg_current_window > avg_previous_window:
                print(
                    f"Fuzzy AUNL: Early stopping triggered at epoch {epoch+1}: Generalization error is increasing."
                )
                self.model.stop_training = True

        print(
            f"Epoch {epoch+1}: AUNL={aunl}, AUNL_VAL={aunl_val}, Generalization error={generalization_error}"
        )

    def sh_stopping(self, epoch):
        if 'trial' in self.args:
            if self.args["trial"].should_prune():
                self.model.stop_training = True
                raise optuna.exceptions.TrialPruned()
        else:
            self.terminate_bad_candidates(epoch)

    
    def terminate_bad_candidates(self, epoch):
        """
        Apply Successive Halving (SH) to remove bad candidates from the shared list.

        Ensures updates apply across all parallel workers.
        """
        print(f"Epoch: {epoch}")
        if not self.should_evaluate(epoch):
            return

        df = pd.DataFrame(list(self.tracker.candidates))  # Convert shared list to DataFrame
        print(f"Candidates: {len(df)}")
        
        # Select the top-performing half based on "metric"
        if df.shape[0] > 2:
            best = df.nsmallest(df.shape[0] // 2, "metric")
        else:
            best = df  # If only 1-2 models exist, keep all

        if "model_id" in self.args:
            if self.args["model_id"] not in best["model_id"].values:
                print(f"Pruning at {epoch+1}: Successive Halving")
                self.model.stop_training = True
                # Modify shared list IN PLACE (required for multiprocessing)

                self.tracker.candidates[:] = [  # Modify in place
                        obj for obj in self.tracker.candidates if obj["model_id"] != self.args["model_id"]
                    ]

                self.tracker.pruned.append(self.args["model_id"])

