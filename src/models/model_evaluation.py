from src.experiments.experiment_param import ExperimentParam
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
import numpy as np
import src.utils.model_calculations as model_calculations


class ModelEvaluation:
    def __init__(self, data, experiment: ExperimentParam, model):
        """
        Initialize the ModelEvaluation class.

        Args:
            data (tuple): A tuple containing features and scaled targets.
            experiment_id (str): A unique identifier for the experiment to load related artifacts.
            model (object): A trained model used for predictions.
        """
        self.experiment = experiment
        self.scaler = self.experiment.scalers["y"]
        self.model = model
        self.X_scaled, self.y_unscaled = self.load_data(data)

    def load_data(self, data):
        """
        Load and transform the test data using the stored scaler.

        Args:
            test_data (tuple): Tuple containing scaled features (X) and scaled targets (y).

        Returns:
            tuple: Scaled features (X_scaled) and unscaled targets (y_unscaled).
        """
        X_scaled, y_scaled = data

        # Ensure y_scaled is reshaped properly before inverse transformation
        y_unscaled = self.scaler.inverse_transform(y_scaled.reshape(-1, 1))

        return X_scaled, y_unscaled


    def evaluate(self, epoch):
        """
        Evaluate the model predictions on the unscaled data.

        Args:
            losses (list or np.array): Training losses per epoch.
            val_losses (list or np.array): Validation losses per epoch.

        Returns:
            dict: A dictionary containing evaluation metrics including MAE, MSE, RMSE, SMAPE, AUNL, and AUNL_VAL.
        """
        y_pred = self.model.predict(self.X_scaled)
        y_pred_unscaled = self.scaler.inverse_transform(y_pred.reshape(-1, 1))

        try:
            # Calculate evaluation metrics on unscaled data
            mae = mean_absolute_error(self.y_unscaled, y_pred_unscaled)
            mse = mean_squared_error(self.y_unscaled, y_pred_unscaled)
            rmse = np.sqrt(mse)

            smape = model_calculations.calculate_smape(
                self.y_unscaled, y_pred_unscaled
            )

            return {"mae": mae, "mse": mse, "rmse": rmse, "smape": smape}
        
        except Exception as e:
            print(f"[ERROR]: {e}")
            print(f"[ERROR]: Returning default values")
            return {"mae": np.inf, "mse": np.inf, "rmse": np.inf, "smape": 2}
