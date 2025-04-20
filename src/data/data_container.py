class DataContainer:
    def __init__(self, train_data=None, val_data=None, test_data=None):
        """
        Initialize the DataContainer with processed data.

        Args:
            train_data (any type, optional): Processed training data. Defaults to None.
            val_data (any type, optional): Processed validation data. Defaults to None.
            test_data (any type, optional): Processed test data. Defaults to None.
        """
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data

    def set_train_data(self, data):
        """
        Set the training data.

        Args:
            data (any type): Processed training data to store.
        """
        self.train_data = data

    def set_val_data(self, data):
        """
        Set the validation data.

        Args:
            data (any type): Processed validation data to store.
        """
        self.val_data = data

    def set_test_data(self, data):
        """
        Set the test data.

        Args:
            data (any type): Processed test data to store.
        """
        self.test_data = data

    def get_train_data(self):
        """
        Get the training data.

        Returns:
            any type: The stored training data.
        """
        return self.train_data

    def get_val_data(self):
        """
        Get the validation data.

        Returns:
            any type: The stored validation data.
        """
        return self.val_data

    def get_test_data(self):
        """
        Get the test data.

        Returns:
            any type: The stored test data.
        """
        return self.test_data

    
    def get_input_shape(self):
        """
        Get the input shape of the training data (X_train) in the format (window_size, num_columns).

        Returns:
            tuple: (window_size, num_columns) or None if training data is not set.
        """
        if not self.train_data:
            return None  # No training data available

        X_train, _ = self.train_data  # Extract X_train
        shape = X_train.shape

        if X_train.ndim == 3:  # Format: (samples, window_size, num_columns)
            _, window_size, num_columns = shape
        elif X_train.ndim == 2:  # Format: (samples, num_columns) → Assume window_size = 1
            window_size, num_columns = shape
        else:
            return None  # Handle unexpected shapes

        return (window_size, num_columns)


    def __str__(self):
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        X_test, y_test = self.test_data

        return (
            f"X_train: {X_train.shape}, y_train: {y_train.shape}, "
            f"X_val: {X_val.shape}, y_val: {y_val.shape}, "
            f"X_test: {X_test.shape}, y_test: {y_test.shape}"
        )
