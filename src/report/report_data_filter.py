import ast
from scipy.stats import gaussian_kde
import numpy as np
import pandas as pd


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """
    Normalize a single array to the range [0, 1].

    Args:
        arr (np.ndarray): The input array to normalize.

    Returns:
        np.ndarray: The normalized array.
    """
    arr_min = arr.min()
    arr_max = arr.max()
    if arr_max - arr_min == 0:
        # Avoid division by zero if all values in the array are the same
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)


def get_whisker_plot_data(data) -> dict:
    """
    Returns a dictionary with metrics as keys and values as lists of lists.
    Each inner list contains metric values for all models at a specific epoch,
    considering only rows where 'evaluation_epoch' is True.


    Returns:
        dict: A dictionary with metrics as keys and lists of lists as values.
    """
    metrics = [
        "mae",
        "mse",
        "rmse",
        "smape",
        "loss",
        "val_loss",
        "aunl",
        "aunl_val",
    ]

    # Filter the data to include only rows where 'evaluation_epoch' is True
    filtered_data = data[data["evaluation_epoch"] == True]

    # Group by epoch and collect metric values
    metrics_by_epoch = {
        metric: [
            group[metric].tolist()
            for _, group in filtered_data.groupby("epoch", sort=True)
        ]
        for metric in metrics
    }

    return metrics_by_epoch

def get_normalized_loss(data, loss_name, model_id: int) -> dict:
    """
    Normalize 'loss' and 'val_loss' columns to the [0, 1] scale for a specific model.

    Args:
        model_id (int): The ID of the model to normalize the loss values.

    Returns:
        dict: A dictionary with normalized 'loss' and 'val_loss' values rounded to 4 decimals.
    """
    # Load data into pandas DataFrame
    df = data[(data["model_id"] == model_id)]

    # Check if 'loss' and 'val_loss' columns exist
    if loss_name not in df.columns:
        raise ValueError("The file must contain 'loss' and 'val_loss' columns.")

    losses = df[loss_name].values

    # Reshape back to original structure
    normalized_losses = normalize_array(losses)

    return normalized_losses

def get_aggregated_loss(data, loss_name, stat_func: str = "mean") -> dict:
    """
    Calculate the mean, median, or mode of normalized loss and val_loss across all models.

    Args:
        stat_func (str): 'mean', 'median', or 'mode' to determine the statistic function to use.

    Returns:
        dict: A dictionary with the calculated mean, median, or mode values for normalized loss and val_loss.
    """
    # Get file paths of all history files
    model_ids = data["model_id"].unique()

    # Get the normalized loss and val_loss for each file using map
    normalized_losses = list(
        map(
            lambda model_id: get_normalized_loss(data, loss_name, model_id),
            model_ids,
        )
    )

    # Find the maximum length of all loss sequences
    max_length = max(len(loss) for loss in normalized_losses)

    # Pad the loss sequences with NaN to align their lengths
    padded_losses = np.array([
        np.pad(loss, (0, max_length - len(loss)), constant_values=np.nan)
        for loss in normalized_losses
    ])

    # Calculate the mean, median, or mode across all models for loss and val_loss
    if stat_func == "mean":
        aggregated_loss = np.mean(padded_losses, axis=0)
    elif stat_func == "median":
        aggregated_loss = np.median(padded_losses, axis=0)
    elif stat_func == "mode":
        aggregated_loss = np.apply_along_axis(kde_mode, 0, padded_losses)
    else:
        raise ValueError("stat_func must be either 'mean', 'median', or 'mode'.")

    return aggregated_loss.round(4).tolist()

def kde_mode(data: np.ndarray, granulation: int = 100, epsilon: float = 1e-6) -> float:
    """
    Calculates the mode of a dataset using Kernel Density Estimation (KDE), 
    ignoring NaN values in the input data.

    Args:
        data (np.ndarray): A 1D array representing the dataset.
        granulation (int): Number of points in the grid for evaluating the KDE.
        epsilon (float): Small noise added to avoid singular covariance issues.

    Returns:
        float: The mode of the dataset based on KDE or the unique value if the data is degenerate.
    """
    if len(data) == 0 or np.all(np.isnan(data)):
        raise ValueError("Input data array is empty or contains only NaNs.")

    # Remove NaN values from the data
    data = data[~np.isnan(data)]

    if len(data) == 0:
        raise ValueError("After removing NaN values, no data remains.")

    data = np.asarray(data, dtype=float)

    # If all values are the same, return that value
    if np.ptp(data) == 0:
        return data[0]

    # Add small noise to avoid singular covariance issues
    data += np.random.normal(0, epsilon, size=data.shape)

    # Define a grid for KDE evaluation
    grid = np.linspace(data.min(), data.max(), granulation)
    kde = gaussian_kde(data)
    density = kde(grid)

    # Find the grid point with the highest density
    mode_index = np.argmax(density)

    return grid[mode_index]


def get_last_epoch(df) -> pd.DataFrame:
    
    filtered_data = df[df["evaluation_epoch"] == True]
    
    ids = df['model_id'].unique()
    arr = []
    
    # Loop through each model_id
    for id in ids:
        # Filter the data for the current model_id and get the row with the maximum epoch
        model_data = filtered_data[filtered_data['model_id'] == id]
        max_epoch_row = model_data.loc[model_data['epoch'].idxmax()]  # Get the row with the max epoch
    
        arr.append(max_epoch_row)
    
    result_df = pd.DataFrame(arr)
    return result_df

def pad_with_nan(data, target_length):
    """
    Pads a list with NaN values until it reaches the target length.
    
    Args:
        data (list): The data list to be padded.
        target_length (int): The target length to pad the data to.
    
    Returns:
        list: The padded list with NaN values.
    """
    return data + [np.nan] * (target_length - len(data))

def get_tabular_data(data) -> list:
    """
    Extracts metrics for best, worst, and aggregated models and prepares them for a template.
    
    Returns:
        dict: A dictionary containing metrics for the best, worst, mean, median, and mode models.
    """
    # Filter the data to find the best model based on minimum 'aunl'
    df_cleaned = data[~data.isin([np.inf, -np.inf]).any(axis=1)]
    filtered_data = df_cleaned[df_cleaned["evaluation_epoch"] == True]

    last_epochs = get_last_epoch(df_cleaned)
    best_model_id = last_epochs.loc[last_epochs["mse"].idxmin()]['model_id']
    worst_model_id = last_epochs.loc[last_epochs["mse"].idxmax()]['model_id']

    metrics = ["mae", "mse", "rmse", "smape", "aunl_val", "training_time"]

    def extract_metrics(data: pd.DataFrame, metric: str, model_id: int) -> list:
        return list(data.loc[data["model_id"] == model_id, metric])

    # Mean model metrics
    mean_model = {
        metric: list(filtered_data.groupby("epoch")[metric].mean().round(4))
        for metric in metrics
    }

    # Median model metrics
    median_model = {
        metric: list(filtered_data.groupby("epoch")[metric].median().round(4))
        for metric in metrics
    }

    # Mode model metrics using KDE
    mode_model = {
        metric: [
            kde_mode(
                filtered_data.loc[filtered_data["epoch"] == epoch, metric].dropna()
            ).round(4)
            for epoch in sorted(filtered_data["epoch"].unique())
        ]
        for metric in metrics
    }
    
    # Find the longest length of any metric (to pad shorter lists)
    max_len = max([len(values) for values in mean_model.values()])
    
    best_model = {
        metric: extract_metrics(filtered_data, metric, best_model_id)
        for metric in metrics
    }

    # Pad best model metrics with NaN values to make the lengths consistent
    best_model = {
        metric: pad_with_nan(values, max_len) for metric, values in best_model.items()
    }
    
    worst_model = {
        metric: extract_metrics(filtered_data, metric, worst_model_id)
        for metric in metrics
    }

    # Pad worst model metrics with NaN values to make the lengths consistent
    worst_model = {
        metric: pad_with_nan(values, max_len) for metric, values in worst_model.items()
    }

    # Now, all models (best, worst, mean, median, mode) have consistent lengths
    return {
        "Best": pd.DataFrame(best_model).astype(float),
        "Mode": pd.DataFrame(mode_model).astype(float),
        "Median": pd.DataFrame(median_model).astype(float),
        "Mean": pd.DataFrame(mean_model).astype(float),
        "Worst": pd.DataFrame(worst_model).astype(float)
    }


def get_epochs(data) -> list:
    """
    Returns a sorted list of unique epochs from the dataset, adjusted by +1.

    Returns:
        list: A list of unique epochs.
    """
    filtered_data = data[data["evaluation_epoch"] == True]
    unique_sorted_epochs = sorted(filtered_data["epoch"].unique())
    processed_epochs = [epoch + 1 for epoch in unique_sorted_epochs]
    return processed_epochs

def get_experiment_parameters(data) -> dict:
    """
    Extracts experiment-related parameters and ensures correct data types.

    Returns:
        dict: A dictionary containing experiment parameters like epochs, patience, step_check, etc.
    """
    experiment = data.iloc[0]
    return {
        "epochs": str(experiment["num_epochs"]),   # Ensure integer
        "patience": str(experiment["patience"]),
        "step_check": str(experiment["step_check"]),
        "train_size": str(experiment["train_size"]),  # Ensure float for sizes
        "val_size": str(experiment["val_size"]),
        "window_size": str(experiment["window_size"]),
        "hpo": str(experiment["hpo"]),   # Convert categorical or object-type data to string
    }

def get_hpo_algorithm_params(data) -> dict:
    """
    Extracts HPO algorithm-related parameters and ensures correct data types.

    Returns:
        dict: A dictionary with optimizer parameters.
    """
    optimizer = data.iloc[0]
    
    if optimizer["hpo"] == "search":
        return {
            "sampler": str(optimizer["search_sampler"]),
            "n_trials": str(optimizer["n_trails"]),
        }

    if optimizer["hpo"] == "differential":
        return {
            "algorithm": "Differential Evolution",
            "strategy": str(optimizer["strategy"]),
            "popsize": str(optimizer["popsize"]),
            "maxiter": str(optimizer["maxiter"]),
            "crossover": str(optimizer["crossover"]),
            "mutation": str(optimizer["mutation"]),
            "init": str(optimizer["init"]),
            "polish": str(optimizer["polish"]),
            "tol": str(optimizer["tol"]),
            "atol": str(optimizer["atol"]),
        }

    return {
        "algorithm": "SimpleEA",
        "population_size": str(optimizer["population_size"]),
        "generations": str(optimizer["generations"]),
        "crossover": str(optimizer["crossover"]),
        "mutation": str(optimizer["mutation"]),
        "indpb": str(optimizer["indpb"]),
        "tournsize": str(optimizer["tournsize"]),
    }

def calculate_total_time(data):
    """
    Calculates the total time taken for the experiment, considering gaps between rows. 
    Gaps larger than 20 minutes are considered as a stop in training, and their time difference is included.
    
    Args:
        data (pd.DataFrame): The dataset containing the timestamps of the epochs.

    Returns:
        str: The total time formatted as HH:MM:SS, where hours can exceed 24.
    """
    # Convert the 'timestamp' column to datetime safely
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors='coerce')

    # Sort by timestamp to ensure correct time order
    data = data.sort_values("timestamp")

    # Calculate the time difference between consecutive rows
    data['time_diff'] = data['timestamp'].diff()

    # Set the maximum tolerated gap (20 minutes)
    max_gap = pd.Timedelta(minutes=20)

    # Filter out rows where the time gap is less than the max tolerated gap
    valid_time_diff = data['time_diff'] <= max_gap

    # Initialize total_time as 0 and accumulate time differences
    total_time = pd.Timedelta(0)
    
    # Loop over the rows and accumulate time difference if it's valid
    for i in range(1, len(data)):
        if valid_time_diff.iloc[i]:
            total_time += data['time_diff'].iloc[i]

    # Convert total time to total seconds to avoid 24-hour overflow issues
    total_seconds = total_time.total_seconds()

    # Calculate hours, minutes, and seconds
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)

    # Return formatted time as HH:MM:SS where hours can exceed 24
    return f"{hours:02}:{minutes:02}:{seconds:02}"


def get_experiment_summary(data):
    """
    Calculates and returns a summary of the experiment's budget details and time duration,
    accounting for gaps between epochs (ignoring small gaps).

    Returns:
        dict: A dictionary containing:
            - 'budget': The budget used for the experiment.
            - 'total_epochs': The total number of epochs for the experiment.
            - 'strategy': The budget strategy employed.
            - 'total_time': The total time taken for the experiment (formatted as a string).
            - 'total_models': The total number of models.
    """
    # Calculate total time using the helper function
    total_time = calculate_total_time(data)

    # Calculate total epochs per model (avoiding NoneType issues)
    max_epoch_rows = data.loc[data.groupby("model_id")["epoch"].idxmax()]
    max_epoch_rows = max_epoch_rows[["epoch"]] + 1  # Adding 1 to account for the last epoch
    epochs = int(max_epoch_rows["epoch"].sum())

    # Ensure budget and strategy fields are properly formatted
    budget_value = data["budget"].loc[0]
    budget = int(budget_value) + 1 if pd.notna(budget_value) else "Non-budget experiment"

    strategy_value = data["budget_strategy"].loc[0]
    strategy = str(strategy_value) if pd.notna(strategy_value) else "Non-budget experiment"

    return {
        "budget": str(budget),
        "total_epochs": str(epochs),
        "strategy": str(strategy),
        "total_time": str(total_time),
        "total_models": str(len(data['model_id'].unique()))
    }

    
def extract_model_params(data, model_id):
    """
    Extracts model-specific parameters for a given model_id.

    This method retrieves the first row of data for the given model_id and extracts key model parameters
    such as the number of layers, neurons, activation function, dropout rate, learning rate, optimizer,
    and batch size.

    Args:
        model_id (int): The ID of the model whose parameters are to be extracted.

    Returns:
        dict: A dictionary containing the model's parameters:
            - 'num_layers': Number of layers in the model.
            - 'layers': The architecture of the layers.
            - 'neurons': The number of neurons in the model.
            - 'activation': The activation function used.
            - 'dropout_rate': The dropout rate of the model.
            - 'learning_rate': The learning rate used for training.
            - 'optimizer': The optimizer used for training.
            - 'batch_size': The batch size used during training.
    """
    # Filter the DataFrame by model_id
    model_data = data[data["model_id"] == model_id].iloc[
        0
    ]  # Get the first row for the model_id

    # Extract the required columns and store them in a dictionary
    model_params = {
        "model_id": str(model_id),
        "num_layers": str(model_data["num_layers"]),
        "layers": str(model_data["layers"]),
        "neurons": str(model_data["neurons"]),
        "activation": str(model_data["activation"]),
        "dropout_rate": str(model_data["dropout_rate"]),
        "learning_rate": str(model_data["learning_rate"]),
        "optimizer": str(model_data["optimizer"]),
        "batch_size": str(model_data["batch_size"]),
    }

    return model_params


def get_epoch_frequency(data):
    """
    Returns the frequency distribution of model epochs.

    Returns:
        pd.DataFrame: A DataFrame containing the frequency of the epoch values,
                    with the epoch adjusted by +1 for proper scaling.
    """
    histogram_data = data.loc[data.groupby("model_id")["epoch"].idxmax()]
    histogram_data = histogram_data[["epoch"]] + 1
    return histogram_data

def get_model_based_on_performance(data, metric, best):
    """
    Selects the model with the maximum or minimum performance based on a specific metric 
    at the last epoch for each model.

    Args:
        data (pandas.DataFrame): The input DataFrame containing model performance data.
            It should have columns such as 'model_id', 'epoch', and the performance metric to be evaluated.
        metric (str): The name of the metric (column) to evaluate in the dataset.
            This should be a string representing the name of a column in `data`.
        max (bool): If `True`, the model with the maximum value for the given metric is selected.
                    If `False`, the model with the minimum value for the given metric is selected.

    Returns:
        str: The model ID of the selected model.
    """
    last_epoch_data = data.loc[data.groupby("model_id")["epoch"].idxmax()]
    if best:
        model = last_epoch_data.loc[last_epoch_data[metric].idxmin()]
    else:
        model = last_epoch_data.loc[last_epoch_data[metric].idxmax()]
    return model["model_id"]
