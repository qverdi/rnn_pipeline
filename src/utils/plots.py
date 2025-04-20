import matplotlib.pyplot as plt
from typing import List, Dict
import seaborn as sns
import numpy as np
import io


def plot_whisker_plots(
    metrics_data: Dict[str, List[float]], epochs: List[int]
) -> None:
    """
    Plots a set of whisker plots for different metrics (e.g., MAE, MSE, RMSE) across epochs.

    Args:
        metrics_data (dict): A dictionary where keys are metric names and values are lists of values for each metric.
        epochs (list[int]): List of epoch numbers to display on the x-axis.
        filename (str): The path where the plot will be saved.

    Returns:
        None
    """
    metric_names = list(metrics_data.keys())
    rows = len(metric_names) // 3 + (1 if len(metric_names) % 3 else 0)
    cols = 3

    fig, axes = plt.subplots(rows, cols, figsize=(2 * len(epochs), 5 * rows))
    axes = axes.flatten()

    for i, metric in enumerate(metric_names):
        data = metrics_data[metric]
        axes[i].boxplot(data)
        axes[i].set_title(f"Whisker Plot - {metric.upper()}")
        axes[i].set_xticklabels([f"{epoch}" for epoch in epochs[: len(data)]])
        axes[i].set_ylabel("Values")

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout(pad=2.0)
    return fig


def align_to_epochs(values: List[float], epochs: List[int]) -> List[float]:
    """
    Aligns a list of model values to the length of the epochs list.
    Pads with NaN if values are shorter than epochs, or truncates if values are longer.
    
    Args:
        values (list[float]): Model values to be aligned.
        epochs (list[int]): The reference list of epochs.
    
    Returns:
        list[float]: Aligned model values, padded or truncated to match the length of epochs.
    """
    len_values = len(values)
    total_epochs = max(epochs)
    
    if len_values < total_epochs:
        # Pad with NaN (using np.nan) if values are shorter
        padded = np.full(total_epochs, np.nan)
        padded[:len_values] = values
        print(len(padded))
        return padded
    return values


def plot_model_comparison(
    best_values: List[float],
    other_values: List[List[float]],
    epochs: List[int],
    model_labels: List[str]
) -> plt.Figure:
    """
    Plots a comparison of the best model and other models over the epochs.
    
    Args:
        best_values (list[float]): Values for the best model over the epochs.
        other_values (list[list[float]]): List of values for other models.
        epochs (list[int]): List of epochs.
        model_labels (list[str]): Labels for the models to be compared.

    Returns:
        plt.Figure: The figure containing the plot.
    """
   
    # Align the best model and other models to the length of epochs
    best_values = align_to_epochs(best_values, epochs)
    aligned_other_values = [align_to_epochs(model, epochs) for model in other_values]

    # Create the figure and axes for subplots
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()  # Flatten the axes array to make it easier to loop through

    # Loop through each model and plot the comparison
    for i, other_value in enumerate(aligned_other_values):
        ax = axes[i]

        # Plot the best model
        ax.plot(range(1, max(epochs) + 1), best_values, label="Best Model", color="green", linewidth=2)

        # Plot the current comparison model (Mode, Median, Mean, etc.)
        ax.plot(range(1, max(epochs) + 1), other_value, label=f"{model_labels[i]}", color="blue", linewidth=2)

        # Fill the area under the curves with color
        ax.fill_between(range(1, max(epochs) + 1), other_value, color="blue", alpha=0.3)
        ax.fill_between(range(1, max(epochs) + 1), best_values, color="green", alpha=0.3)

        # Set labels, title, and grid for better readability
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Model Values")
        ax.set_title(f"Best vs {model_labels[i]} Performance")
        ax.set_xticks(epochs)  # Ensure x-ticks correspond to epochs
        ax.grid(True, linestyle="-", alpha=0.7)
        ax.legend()

    # Adjust layout to make the plot more compact
    plt.tight_layout()

    return fig


def plot_model_performance(
    values: List[float], epochs: List[int], title: str, filename: str
) -> None:
    """
    Plots the performance of a model over epochs.

    Args:
        values (list[float]): The model's performance values over the epochs.
        epochs (list[int]): List of epochs.
        title (str): The title of the plot.
        filename (str): The path where the plot will be saved.

    Returns:
        None
    """
    fig = plt.figure(figsize=(10, 6))
    plt.grid(True, linestyle="-", alpha=0.3)

    plt.plot(
        range(len(values)), values, label="Model Performance", color="blue", linewidth=2
    )
    plt.fill_between(range(len(values)), values, color="blue", alpha=0.3)

    for epoch in epochs:
        if epoch < len(values):
            plt.plot(
                [epoch, epoch],
                [0, values[epoch]],
                color="blue",
                linestyle=":",
                linewidth=1.5,
            )

    plt.xlabel("Epochs")
    plt.ylabel("Model Values")
    plt.title(title)
    plt.xticks(range(len(values)))
    plt.legend()
    plt.tight_layout()
    plt.grid(True, linestyle="-", alpha=0.7)
    return fig



def plot_metric_comparison(
    best_model: Dict[str, List[float]],
    mode_model: Dict[str, List[float]],
    median_model: Dict[str, List[float]],
    mean_model: Dict[str, List[float]],
    worst_model: Dict[str, List[float]],
    epochs: List[int],
    metrics: List[str],
) -> None:
    """
    Plots a comparison of different metrics across various models over epochs.

    Args:
        best_model (dict): Metrics for the best model.
        mode_model (dict): Metrics for the mode model.
        median_model (dict): Metrics for the median model.
        mean_model (dict): Metrics for the mean model.
        worst_model (dict): Metrics for the worst model.
        epochs (list[int]): List of epochs.
        metrics (list[str]): List of metrics to compare (e.g., 'mae', 'mse').
        filename (str): The path where the plot will be saved.

    Returns:
        None
    """
    models = {
        "Best Model": best_model,
        "Mode Model": mode_model,
        "Median Model": median_model,
        "Mean Model": mean_model,
        "Worst Model": worst_model,
    }

    num_metrics = len(metrics)
    rows = 2
    cols = (num_metrics + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 6 * rows), sharex=True)
    axes = axes.flatten() if num_metrics > 1 else [axes]

    for i, metric_name in enumerate(metrics):
        ax = axes[i]

        for model_name, model_data in models.items():
            if metric_name in model_data:
                values = model_data[metric_name]
                ax.plot(range(len(values)), values, label=model_name, linewidth=2)

        ax.set_xlabel("Epochs")
        ax.set_ylabel(metric_name.capitalize())
        ax.set_title(f"Comparison of {metric_name.capitalize()}")
        ax.set_xticks(range(len(best_model[metric_name])))
        ax.set_xticklabels(epochs[: len(best_model[metric_name])])
        ax.grid(True, linestyle="-", alpha=0.7)
        ax.legend(loc="upper right")

    for j in range(len(metrics), len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    return fig


def plot_aunl_comparison(epochs, model1, model2, total_epochs=50):
    # Ensure both model1 and model2 are the same length as total_epochs
    len_model1 = len(model1)
    len_model2 = len(model2)

    # Create an array with NaN values for total_epochs length
    full_model1 = np.full(total_epochs, np.nan)
    full_model2 = np.full(total_epochs, np.nan)

    # Align model1 and model2 values to the full range of total_epochs
    full_model1[:len_model1] = model1
    full_model2[:len_model2] = model2

    # Create the plot with custom line colors and surface fill
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the lines with custom colors
    ax.plot(range(1, total_epochs + 1), full_model1, label="Model 1", color="blue", linewidth=2)
    ax.plot(range(1, total_epochs + 1), full_model2, label="Model 2", color="green", linewidth=2)

    # Fill the area under the lines with surface color
    ax.fill_between(range(1, total_epochs + 1), full_model1, color="blue", alpha=0.3)
    ax.fill_between(range(1, total_epochs + 1), full_model2, color="green", alpha=0.3)

    # Set y-axis limits from 0 to 1
    ax.set_ylim(0, 1)

    # Mark x labels by epochs
    ax.set_xticks(epochs)  # Set x-axis labels to epochs
    ax.set_xticklabels(epochs, rotation=45)  # Optional: Rotate the labels for better readability

    # Adding labels, title, and grid
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Model 1 vs Model 2")
    ax.legend()
    ax.grid(True)

    return fig

def plot_epoch_frequency(data):
    """
    Plots the frequency of epochs from the provided dataset and saves the plot to a file.

    This function takes a series of epoch data, calculates the frequency of each unique total epochs per model, and 
    creates a bar plot to visualize the distribution of epochs. The plot is then saved to the specified 
    filename.

    Args:
        data (pd.Series): A pandas Series containing epoch data. Each entry represents an epoch.
    """    
    # Create data for the barplot (count of each unique epoch)
    epoch_counts = data.value_counts().sort_index()

    # Ensure the index is flat (not a MultiIndex) by resetting it if necessary
    epoch_counts = epoch_counts.reset_index()
    epoch_counts.columns = ['epoch', 'frequency']  # Rename columns for clarity

    # Adjust plot size based on the number of unique epochs
    plot_width = max(5, len(epoch_counts) * 0.5)  # Ensure a minimum width
    plot_height = 5  # Fixed height, adjust as needed
    fig = plt.figure(figsize=(plot_width, plot_height))

    # Create the bar plot
    ax = sns.barplot(x='epoch', y='frequency', data=epoch_counts, color='skyblue', edgecolor='black')

    # Add values on top of the bars as integers
    for p in ax.patches:
        height = p.get_height()  # Get the height of the bar
        ax.text(p.get_x() + p.get_width() / 2, height + 0.1,  # Place the text slightly above the bar
                int(height),  # Display the value as integer
                ha='center', va='bottom', fontsize=10)  # Align and format the text

    # Set labels and title
    plt.xlabel('Epochs')
    plt.ylabel('Frequency')
    plt.title('Frequency of Each Epoch')
    plt.tight_layout()  # Adjust layout for better spacing

    return fig

def get_svg_from_figure(fig):
    img_buf = io.BytesIO()
    fig.savefig(img_buf, format="svg")
    plt.close(fig)  # Close the figure to free memory
    return img_buf.getvalue().decode("utf-8")  # Decode bytes to string