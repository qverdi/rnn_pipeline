import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from src.experiments.experiment_param import ExperimentParam


class DataProcessing:
    def __init__(self, data: pd.DataFrame, experiment_params: ExperimentParam):
        self.data = data
        self.experiment_params = experiment_params
        self.train_data, self.val_data, self.test_data = self._split_data(data)
        self.scalers = {}  # Initialize scaler, will be set during training scaling

    def _split_data(self, data):
        """
        Split data into train, validation, and test sets based on time series index.
        Returns NumPy arrays instead of Pandas DataFrames.
        """
        train_size = int(len(data) * self.experiment_params.train_size)
        val_size = int(len(data) * self.experiment_params.val_size)

        # Convert DataFrame slices to NumPy arrays
        return (
            data.iloc[:train_size].to_numpy(),
            data.iloc[train_size : train_size + val_size].to_numpy(),
            data.iloc[train_size + val_size :].to_numpy(),
        )

    def _shapiro_wilk_test(self, data):
        """
        Perform the Shapiro-Wilk test for normality on the data.
        Returns True if data is normally distributed, otherwise False.
        """
        stat, p_value = stats.shapiro(data)
        return p_value > 0.05  # Null hypothesis: Data is normal if p > 0.05

    def _scale_data(self, data, scaler_name, fit=True):
        """
        Scale data using StandardScaler or MinMaxScaler based on normality test.
        Handles both 2D (targets) and 3D (features) data.

        Args:
            data (np.ndarray or pd.DataFrame): Data to scale.
            scaler_name (str): Unique name for the scaler (e.g., 'X', 'Y').
            fit (bool): Whether to fit the scaler (only on training data).

        Returns:
            np.ndarray: Scaled data (same shape as input).
        """
        # Initialize scaler if not already set
        if scaler_name not in self.scalers:
            self.scalers[scaler_name] = StandardScaler() if self._shapiro_wilk_test(data) else MinMaxScaler()
            print(f"Using {type(self.scalers[scaler_name]).__name__} for {scaler_name}")

        scaler = self.scalers[scaler_name]
        scaled_data = scaler.fit_transform(data) if fit else scaler.transform(data)

        # Store the scaler only when fitting
        if fit:
            self.experiment_params.scalers[scaler_name] = scaler

        return scaled_data


    def process_data(self):
        """
        Process data: create sequences, scale data, and return scaled datasets.
        """

        def _prepare_sequences(data, fit):
            """
            Creates time-series sequences, reshapes for scaling, applies scaling, and restores shape.

            Args:
                data (np.ndarray or pd.DataFrame): Input time-series data.
                fit (bool): Whether to fit the scaler (only for training data).

            Returns:
                tuple: Scaled input sequences (X) and scaled target values (y).
            """
            # Generate sequences for input (X) and target (y)
            X, y = create_sequences(
                data, 
                self.experiment_params.window_size, 
                self.experiment_params.target_column
            )

            # Store original shape of X
            samples, window_size, num_features = X.shape  

            # Reshape X to 2D for scaling and then reshape it back to 3D
            X_scaled = self._scale_data(X.reshape(-1, num_features), "X", fit)
            X_scaled = X_scaled.reshape(samples, window_size, num_features)

            # Reshape y to 2D for scaling
            y_scaled = self._scale_data(y.reshape(-1, 1), "y", fit)

            return X_scaled, y_scaled

        # Process train, validation, and test datasets
        train_set = _prepare_sequences(self.train_data, fit=True)
        val_set = _prepare_sequences(self.val_data, fit=False)
        test_set = _prepare_sequences(self.test_data, fit=False)

        return train_set, val_set, test_set



def create_sequences(data, window_size, target_column=0):
    """
    Generate sequences from time series data, handling both single and multi-column cases.

    Args:
        data (pd.DataFrame or np.ndarray): Time series data with one or more columns.
        window_size (int): Number of past timesteps to include in each sequence.

    Returns:
        tuple:
            - X (np.ndarray): Shape (samples, window_size, num_features).
            - y (np.ndarray): Shape (samples, num_features).
    """
    X, y = [], []

    for i in range(len(data) - window_size):
        X.append(data[i : i + window_size])  # Sequence of past `window_size` timesteps
        y.append(data[i + window_size, target_column].reshape(-1, 1))

    # Convert lists to NumPy arrays
    X, y = np.array(X), np.array(y)
    return X, y
