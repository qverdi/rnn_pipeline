import os
import pandas as pd
from src.config.file_constants import README_FILE

def load_experiment_data(folder_path):
        """
        Load CSV files starting with 'e_' from the given folder into a dictionary.

        Args:
            folder_path (str): The path to the folder containing CSV files.

        Returns:
            dict: A dictionary where keys are experiment IDs and values are DataFrames.
        """
        experiment_data = {}

        # Iterate through all files in the folder
        for filename in os.listdir(folder_path):
            if filename.startswith('e_') and filename.endswith('.csv'):
                # Extract experiment ID from the filename (e_{id}.csv)
                experiment_id = filename.split('_')[1].split('.')[0]
                
                # Construct full file path
                file_path = os.path.join(folder_path, filename)
                
                # Read CSV file into a DataFrame
                df = pd.read_csv(file_path)

                # Add DataFrame to dictionary with experiment ID as key
                experiment_data[experiment_id] = df

        return experiment_data


def get_experiment_ids(folder_path):
    """
    Retrieve a list of experiment IDs from CSV filenames in the given folder.

    Args:
        folder_path (str): The path to the folder containing CSV files.

    Returns:
        list: A list of experiment IDs extracted from filenames.
    """
    experiment_ids = []

    # Iterate through all files in the folder
    for filename in os.listdir(folder_path):
        if filename.startswith('e_') and filename.endswith('.csv'):
            # Extract experiment ID from the filename (e_{id}.csv)
            experiment_id = filename.split('_')[1].split('.')[0]
            experiment_ids.append(experiment_id)

    return experiment_ids


def update_readme(branch_name):
    """
    Updates the last line of the README file to indicate the current experiment branch.

    Args:
        branch_name (str): The name of the experiment branch.
        file_path (str): Path to the README file. Defaults to "README.md".
    """
    try:
        with open(README_FILE, "r", encoding="utf-8") as file:
            lines = file.readlines()

        new_last_line = f"💡 Currently showing results from: `{branch_name}`\n"

        if lines:
            lines[-1] = new_last_line  # Replace the last line
        else:
            lines.append(new_last_line)  # If file is empty, just add the line

        with open(README_FILE, "w", encoding="utf-8") as file:
            file.writelines(lines)

        print(f"✅ README updated with branch: {branch_name}")

    except Exception as e:
        print(f"❌ Error updating README: {e}")