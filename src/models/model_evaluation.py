from src.experiments.experiment_param import ExperimentParam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import src.utils.model_calculations as model_calculations


class ModelEvaluation:
    def __init__(self, data, experiment: ExperimentParam, model):
        """
        Args:
            data (tuple): (X_scaled, y_scaled)
            experiment (ExperimentParam): Holds scalers and config
            model (object): Trained TensorFlow/Keras model
        """
        self.experiment = experiment
        self.scaler = self.experiment.scalers["y"]
        self.model = model
        self.X_scaled, self.y_unscaled = self.load_data(data)

    def load_data(self, data):
        """
        Inverse-transform y to unscaled (original) values.

        Returns:
            tuple: (X_scaled, y_unscaled) with y_unscaled in shape (samples, horizon)
        """
        X_scaled, y_scaled = data

        # Ensure y_scaled is 2D: (samples, horizon)
        if y_scaled.ndim == 3:
            y_scaled = y_scaled.squeeze(-1)

        # Inverse-transform directly in 2D
        y_unscaled = self.scaler.inverse_transform(y_scaled)

        return X_scaled, y_unscaled


    def evaluate(self, epoch):
        """
        Evaluate model performance using unscaled predictions and ground truth.
        Works for multi-step forecasting with shape (samples, horizon).
        """
        y_pred_scaled = self.model.predict(self.X_scaled)

        # If prediction is (samples, horizon, 1), squeeze to (samples, horizon)
        if y_pred_scaled.ndim == 3 and y_pred_scaled.shape[2] == 1:
            y_pred_scaled = y_pred_scaled.squeeze(-1)

        # 👇 This is the KEY change
        y_pred_unscaled = self.scaler.inverse_transform(y_pred_scaled)

        # Ensure y_true is also 2D
        if self.y_unscaled.ndim == 3:
            y_true_unscaled = self.y_unscaled.squeeze(-1)
        else:
            y_true_unscaled = self.y_unscaled

        # Safety check
        if y_pred_unscaled.shape != y_true_unscaled.shape:
            raise ValueError(f"[SHAPE MISMATCH] y_pred: {y_pred_unscaled.shape} ≠ y_true: {y_true_unscaled.shape}")

        try:
            # Flatten for metric computation
            y_true_flat = y_true_unscaled.reshape(-1)
            y_pred_flat = y_pred_unscaled.reshape(-1)

            mae = mean_absolute_error(y_true_flat, y_pred_flat)
            mse = mean_squared_error(y_true_flat, y_pred_flat)
            rmse = np.sqrt(mse)
            smape = model_calculations.calculate_smape(y_true_flat, y_pred_flat)

            return {
                "mae": mae,
                "mse": mse,
                "rmse": rmse,
                "smape": smape,
            }

        except Exception as e:
            print(f"[ERROR]: {e}")
            return {"mae": np.inf, "mse": np.inf, "rmse": np.inf, "smape": 2}
