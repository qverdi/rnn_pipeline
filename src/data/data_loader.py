import pandas as pd
import os
from src.experiments.experiment_param import ExperimentParam


class DataLoader:
    def __init__(self, experiment: ExperimentParam):
        """
        Initializes the DataLoader by loading data from the specified path.

        Args:
            experiment (ExperimentParam): Experiment parameters containing file path and settings.
        """
        self.experiment = experiment
        self.data = self._load_data()

    def _load_data(self):
        """
        Private method to load data based on the file extension.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            ValueError: If the file extension is not supported.
        """
        # Extract file extension
        _, ext = os.path.splitext(self.experiment.file_path)

        # Load data based on file extension
        if ext.lower() == ".csv":
            data = self._load_csv()
        elif ext.lower() in [".xls", ".xlsx"]:
            data = self._load_excel()
        elif ext.lower() == ".txt":
            data = self._load_txt()
        else:
            raise ValueError(f"Unsupported file type: {ext}")

        df = pd.DataFrame(data)
        return df

    def _has_headers(self, df):
        """
        Determines if the first row of the DataFrame contains headers.

        Args:
            df (pd.DataFrame): DataFrame loaded without specifying headers.

        Returns:
            bool: True if the first row contains strings (headers), False otherwise.
        """
        first_row = df.iloc[0]
        return all(isinstance(value, str) for value in first_row)

    def _load_csv(self):
        """Loads a CSV file."""
        header = (
            0
            if self._has_headers(
                pd.read_csv(self.experiment.file_path, nrows=1, header=None)
            )
            else None
        )
        return pd.read_csv(self.experiment.file_path, header=header)

    def _load_excel(self):
        """Loads an Excel file."""
        return pd.read_excel(self.experiment.file_path)

    def _load_txt(self):
        """Loads a TXT file (assumes tab-separated by default)."""
        header = (
            0
            if self._has_headers(
                pd.read_csv(
                    self.experiment.file_path, delimiter="\t", nrows=1, header=None
                )
            )
            else None
        )
        return pd.read_csv(self.experiment.file_path, delimiter="\t", header=header)

    def get_data(self):
        """
        Returns the loaded data.

        Returns:
            pd.DataFrame: The loaded data.
        """
        return self.data
