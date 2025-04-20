from src.utils.json_utils import get_value_from_key
from src.config.file_constants import EXPERIMENT_PARAMS
import tensorflow as tf


def configure_gpu():
    """
    Configures TensorFlow to use a specific GPU if available and specified.
    Falls back to CPU if no GPUs are detected or if the configuration is invalid.

    The GPU to use is fetched from the EXPERIMENT_PARAMS JSON file using the `get_value_from_key` function.
    """
    try:
        # Retrieve the GPU index from experiment parameters
        gpu_unit = get_value_from_key(EXPERIMENT_PARAMS, "gpu")

        # List available GPUs
        gpus = tf.config.list_physical_devices("GPU")

        if not gpus:
            # No GPUs detected, fall back to CPU
            print("No GPUs detected. TensorFlow will run on CPU.")
            return

        # If GPU index is specified, validate it
        if gpu_unit is not None:
            if gpu_unit < 0 or gpu_unit >= len(gpus):
                print(
                    f"Invalid GPU index {gpu_unit}. Only {len(gpus)} GPU(s) available. Using default settings."
                )
            else:
                # Set TensorFlow to use the specified GPU
                tf.config.set_visible_devices(gpus[gpu_unit], "GPU")
                # Optional: Enable memory growth for the GPU
                tf.config.experimental.set_memory_growth(gpus[gpu_unit], True)
                print(f"Successfully configured TensorFlow to use GPU {gpu_unit}.")
        else:
            print(
                "No specific GPU index specified. TensorFlow will use default GPU settings."
            )

    except Exception as e:
        print(f"An unexpected error occurred during GPU configuration: {e}")
