from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, GRU, Dense, Dropout
from src.models.model_params import ModelParams
from src.utils.model_calculations import interpolate_weights
import uuid
import hashlib
import numpy as np

class Model:
    def __init__(self, model_params: ModelParams):
        """
        Initializes the model with a given ParameterSet.

        Args:
            parameters (ParameterSet): An instance of ParameterSet containing model parameters.
        """
        self.id = self.id = hashlib.sha256(uuid.uuid4().bytes).hexdigest()[
            :8
        ]  # Generate a unique ID for each parameter set
        self.params = model_params

        self.model = None

    def _get_rnn_layer(self, layer):
        """
        Maps the network type to the corresponding RNN layer class.

        Returns:
            Layer class (SimpleRNN, LSTM, GRU).
        """
        layer_types = {
            "rnn": SimpleRNN,
            "lstm": LSTM,
            "gru": GRU,
        }
        if layer not in layer_types:
            raise ValueError(f"Unsupported network type: {layer}")
        return layer_types[layer]

    def build_model(self):
        """
        Build and compile the model dynamically based on the provided parameters.
        """
        rnn_layer = self._get_rnn_layer(self.params.layers[0])
        model = Sequential()

        # First layer with input shape
        model.add(
            rnn_layer(
                self.params.neurons[0],
                activation=self.params.activation[0],
                return_sequences=(self.params.num_layers > 1),
                input_shape=self.params.input_shape,
            )
        )
        model.add(Dropout(self.params.dropout_rate[0]))

        # Additional layers
        for i in range(1, self.params.num_layers):
            return_sequences = i < self.params.num_layers - 1
            rnn_layer = self._get_rnn_layer(self.params.layers[i])
            model.add(
                rnn_layer(
                    self.params.neurons[i],
                    activation=self.params.activation[i],
                    return_sequences=return_sequences,
                )
            )
            model.add(Dropout(self.params.dropout_rate[i]))

        # Output layer
        model.add(Dense(1))

        # Compile the model
        model.compile(optimizer=self._get_optimizer(), loss=self.params.loss)

        self.model = model

    def _get_optimizer(self):
        """
        Returns optimizer.

        Returns:
            optimizer: Keras model optimizer.
        """    
        return self.params.optimizer_params.get_optimizer(self.params.learning_rate)

    def get_model(self):
        """
        Returns the built model.

        Returns:
            model: Compiled Keras model.
        """
        if self.model is None:
            raise ValueError("Model has not been built yet. Call `build_model` first.")
        return self.model

    def set_model(self, model):
        """
        Sets the model to the provided model.

        Args:
            model: Compiled Keras model.
        """
        self.model = model


    def set_weights(self, weights):
        """
        Sets the model weights, interpolating if needed.

        Args:
            weights (list of np.ndarray): Model weights to set.
        """
        if not weights:
            print("No weights provided.")
            return

        model_weights = self.model.get_weights()  # Get model's current weight shapes

        if not isinstance(weights, list) or len(weights) != len(model_weights):
            print("Provided weights list does not match model weight count or format.")
            return

        new_weights = [
            interpolate_weights(weight, model_weight.shape) if weight.shape != model_weight.shape else weight
            for model_weight, weight in zip(model_weights, weights)
        ]

        self.model.set_weights(new_weights)
        print("Weights loaded successfully.")
