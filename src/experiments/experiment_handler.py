import os
from multiprocessing import Pool
from pathlib import Path
from src.experiments.experiment_param import ExperimentParam
from src.config.file_constants import (
    DATA_DIR,
    EXPERIMENT_PARAMS,
    SEARCH_SPACE_FILE
)
from src.utils import json_utils
from src.optimization.hpo_manager import HPOManager

class ExperimentHandler:
    """
    Handles the lifecycle of experiments, including initialization, loading datasets,
    optimization (both single-process and multi-process), and report updates.

    Attributes:
        datasets (list): List of dataset file paths to be used in the experiments.
        experiments (list): List of `ExperimentParam` instances for each dataset.
    """

    def __init__(self):
        """
        Initializes the ExperimentHandler instance and sets up experiments.
        """
        self.datasets = []
        self.experiments = []
        self.initialize_experiments()

    def initialize_experiments(self):
        """
        Initialize experiments by creating an `ExperimentParam` for each dataset.
        Reads experiment parameters from a JSON file and associates them with datasets.
        Every dataset has same experiment parameters.
        """
        self.load_datasets()
        experiment_params = json_utils.read_json_file(EXPERIMENT_PARAMS)
        self.experiments = list(
            map(
                lambda dataset_path: ExperimentParam(
                    params=experiment_params, file_path=dataset_path
                ),
                self.datasets,
            )
        )

    def load_datasets(self):
        """
        Load all file paths from the `DATA_DIR` directory, filtering for `.txt`, `.csv`,
        or Excel files, and save them into an array.

        Raises:
            FileNotFoundError: If `DATA_DIR` does not exist or is not a directory.
        """
        data_dir_path = Path(DATA_DIR)
        if not data_dir_path.exists() or not data_dir_path.is_dir():
            raise FileNotFoundError(
                f"DATA_DIR '{DATA_DIR}' does not exist or is not a directory."
            )

        # Supported file extensions
        valid_extensions = {".txt", ".csv", ".xls", ".xlsx"}

        # Get all file paths in DATA_DIR with valid extensions
        self.datasets = [
            str(file_path)
            for file_path in data_dir_path.glob("*")
            if file_path.is_file() and file_path.suffix in valid_extensions
        ]

    def conduct(self):
        """
        Conducts optimization for each experiment sequentially using a lambda
        function and map.
        """
        search_space = json_utils.read_json_file(SEARCH_SPACE_FILE)

        # Map each experiment to an HPOManager and execute its optimization
        list(
            map(
                lambda experiment: HPOManager(search_space, experiment).optimize(),
                self.experiments,
            )
        )