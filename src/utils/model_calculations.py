import numpy as np
import scipy.ndimage

def calculate_aunl(losses, val_losses):
    """
    Calculate AUNL (Area Under Normalized Loss) using the provided losses and validation losses.

    Args:
        losses (list or np.array): Training losses per epoch.
        val_losses (list or np.array): Validation losses per epoch.

    Returns:
        tuple: (aunl, aunl_val) where:
            - aunl is the AUNL for training losses.
            - aunl_val is the AUNL for validation losses.
    """

    # Ensure inputs are numpy arrays
    losses = np.array(losses, dtype=np.float64)
    val_losses = np.array(val_losses, dtype=np.float64)

    # ✅ Replace NaNs with inf to indicate divergence
    losses[np.isnan(losses)] = np.inf
    val_losses[np.isnan(val_losses)] = np.inf

    # ✅ If all values are NaN/inf, return safe default AUNL
    if np.any(np.isinf(losses)):
        return 1, 1
    if np.any(np.isinf(val_losses)):
        return 1, 1

    # Number of points
    n = len(losses)
    if n <= 1:
        return 1, 1  # Not enough data to compute AUNL

    # ✅ Handle case where min == max to avoid division by zero
    losses_min, losses_max = np.nanmin(losses), np.nanmax(losses)
    val_losses_min, val_losses_max = np.nanmin(val_losses), np.nanmax(val_losses)

    if losses_max == losses_min:
        losses_scaled = np.ones_like(losses)  # Avoid zero division
    else:
        losses_scaled = (losses - losses_min) / (losses_max - losses_min)

    if val_losses_max == val_losses_min:
        val_losses_scaled = np.ones_like(val_losses)  # Avoid zero division
    else:
        val_losses_scaled = (val_losses - val_losses_min) / (val_losses_max - val_losses_min)

    # Calculate AUNL using trapezoidal rule
    h = 1 / (n - 1)  # Uniform step size
    aunl = np.sum((losses_scaled[:-1] + losses_scaled[1:]) / 2) * h
    aunl_val = np.sum((val_losses_scaled[:-1] + val_losses_scaled[1:]) / 2) * h

    return aunl, aunl_val


def calculate_smape(y_true, y_pred):
    """
    Calculate Symmetric Mean Absolute Percentage Error (SMAPE).

    Args:
        y_true (array-like): Array of true values.
        y_pred (array-like): Array of predicted values.

    Returns:
        float: SMAPE value.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Avoid division by zero
    denominator = np.where(denominator == 0, 1e-10, denominator)

    smape = np.mean(numerator / denominator)
    return smape


import numpy as np
import scipy.ndimage

def interpolate_weights(source, target_shape):
    """
    Interpolates a weight matrix from any shape to the desired target shape.

    Args:
        source (np.array): The source weight matrix (can be 1D or 2D).
        target_shape (tuple): The desired target shape.

    Returns:
        np.array: The resized weight matrix with the target shape.
    """
    # Ensure source is a numpy array
    source = np.asarray(source)

    # ✅ Handle cases where target shape is 1D
    if isinstance(target_shape, int) or len(target_shape) == 1:
        if source.ndim == 2:
            # If source is 2D but target is 1D (flatten it)
            source = source.flatten()
        return np.resize(source, target_shape)

    # ✅ Handle case where target is 2D but source is 1D
    if source.ndim == 1 and len(target_shape) == 2:
        source = source.reshape(1, -1)  # Convert to row vector

    # ✅ If the shapes already match, return as is
    if source.shape == target_shape:
        return source

    # ✅ Handle interpolation for 2D matrices
    return _interpolate_matrix(source, target_shape)


def _interpolate_matrix(source_weight, target_shape):
    """
    Interpolates a weight matrix to match the target shape.

    Args:
        source_weight (np.ndarray): The source weight matrix.
        target_shape (tuple): The desired target shape.

    Returns:
        np.ndarray: The interpolated weight matrix.
    """
    source_shape = source_weight.shape

    # ✅ Case: 1D -> 1D interpolation
    if source_weight.ndim == 1 and len(target_shape) == 1:
        zoom_factor = target_shape[0] / source_weight.shape[0]
        return scipy.ndimage.zoom(source_weight, zoom_factor, order=1)  # Linear interpolation

    # ✅ Case: 2D -> 2D interpolation
    elif source_weight.ndim == 2 and len(target_shape) == 2:
        zoom_factors = np.array(target_shape) / np.array(source_weight.shape)
        return scipy.ndimage.zoom(source_weight, zoom_factors, order=1)

    # ✅ Case: 2D -> 1D (Flatten while keeping data structure intact)
    elif source_weight.ndim == 2 and len(target_shape) == 1:
        interpolated = scipy.ndimage.zoom(source_weight, (target_shape[0] / source_weight.shape[0], 1), order=1)
        return interpolated.flatten()  # Convert back to 1D

    # ✅ Case: 1D -> 2D (Expand to match target)
    elif source_weight.ndim == 1 and len(target_shape) == 2:
        expanded = np.resize(source_weight, target_shape)
        return expanded

    else:
        raise ValueError(f"Cannot interpolate from shape {source_weight.shape} to {target_shape}")
