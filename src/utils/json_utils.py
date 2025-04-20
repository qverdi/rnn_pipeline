import json


def read_json_file(file_path):
    """
    Reads a JSON file from the given file path and returns its contents as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The contents of the JSON file as a dictionary.
    """
    try:
        with open(file_path, "r") as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at path '{file_path}' does not exist.")
    except json.JSONDecodeError:
        raise ValueError(f"The file at path '{file_path}' is not a valid JSON file.")
    except Exception as e:
        raise RuntimeError(f"An error occurred while reading the file: {e}")


def get_value_from_key(path, key):
    """
    Reads a JSON file and returns the value associated with the given key.

    Parameters:
        path (str): The path to the JSON file.
        key (str): The key whose value needs to be retrieved.

    Returns:
        Any: The value associated with the key.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        json.JSONDecodeError: If the file is not a valid JSON.
        KeyError: If the key is not found in the JSON data.
        Exception: For any other unexpected errors.
    """
    try:
        with open(path, "r") as file:
            data = json.load(file)

        if key not in data:
            raise KeyError(f"Key '{key}' not found in the JSON file.")
        return data[key]

    except FileNotFoundError:
        raise FileNotFoundError(f"File not found at path '{path}'.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError("Failed to decode JSON from file.", doc=path, pos=0)
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")


def write_json_file(file_path, data):
    """
    Writes a Python dictionary or list to a JSON file.

    Parameters:
        file_path (str): The path to the JSON file.
        data (dict or list): The data to write to the file.

    Raises:
        FileNotFoundError: If the file path is invalid.
        PermissionError: If the program lacks permissions to write to the file.
        TypeError: If the provided data cannot be serialized to JSON.
        Exception: For any other unexpected errors.
    """
    try:
        with open(file_path, "w") as f:
            json.dump(data, f, indent=4)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file path '{file_path}' does not exist.")
    except PermissionError:
        raise PermissionError(
            f"Permission denied while trying to write to '{file_path}'."
        )
    except TypeError as e:
        raise TypeError(f"Data provided is not serializable to JSON: {e}")
    except Exception as e:
        raise Exception(f"An unexpected error occurred: {e}")
