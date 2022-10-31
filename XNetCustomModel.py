import numpy as np
import tensorflow as tf


class CustomModel(tf.keras.models.Model):

    def __init__(self, inputs, outputs, **kwargs):
        """
        Override keras.Model init by adding watermarking parameters
        """
        super().__init__(inputs=inputs, outputs=outputs, **kwargs)
        self._hosts = None
        self._watermark = None
        self._watermark_chunks = None
        self._watermark_strength = None
        self._watermark_values = None
        self._host_layer_indices = None
        self._host_layer_types = None
        self._disable_embed = False
        self._weight_difference = []
        self._weight_values = []
        self._previous_batch_weight = []
        self._counter = 0
        return

    def GetHosts(self):
        """
        Getter for the map of the weights that host watermark bits
        :return: watermark hosts, list of tuples that are the coordinates of the watermarked weights
        """
        return self._hosts

    def GetWatermark(self):
        """
        Getter for the watermark sequence
        :return: antipodal [-1, +1] Numpy sequence that is embedded into cnn weights
        """
        return self._watermark

    def GetDisableStatus(self):
        """
        Getter for the flag watermark disabled
        :return: True or False
        """
        return self._disable_embed

    def GetWatermarkStrength(self):
        """
        Getter for watermark strength
        :return: Watermark embedding strength
        """
        return self._watermark_strength

    def GetHostLayerIndices(self):
        """
        Getter for the indices of the network layers hosting the watermark
        :return: Array of indices corresponding to the position of each layer in the self.model.weights structure
        """
        return self._host_layer_indices

    def GetHostLayerInstances(self):
        """
        Getter for the instances of the network layers hosting the watermark
        :return: Array whose elements are the instances of each layer hosting the watermark
        """
        return self._host_layer_types

    def GetWatermarkChunks(self):
        """
        Getter for the chunks of the watermark
        :return: Array whose elements are the lengths of the portion of the watermark assigned to each host layer
        """
        return self._watermark_chunks

    def SetParams(self, hosts, watermark, strength, host_layer_indices, host_layer_types, watermark_chunks):
        """
        Initializes watermarking parameters
        :param hosts: watermark hosts, list of tuples that are the coordinates of the watermarked weights
        :param watermark: antipodal [-1, +1] Numpy sequence that is embedded into cnn weights
        :param strength: watermark embedding strength
        :param host_layer_indices: Array of indices of each host layer in self.model.weights list
        :param host_layer_types: Array of layer types for each layer hosting the watermark (eg. Conv2D ..)
        :param watermark_chunks: Array whose elements are the length of the watermark chunk for each host layer
        :return: Nothing
        """
        self._hosts = hosts
        self._watermark = watermark
        self._watermark_chunks = watermark_chunks
        self._watermark_values = watermark * strength
        self._watermark_strength = strength
        self._host_layer_indices = host_layer_indices
        self._host_layer_types = host_layer_types
        self._weight_values = [[] for i in range(len(host_layer_indices))]
        #self._weight_difference = []
        #self._counter = np.zeros(len(host_layer_indices))
        #for c in np.abs(watermark_chunks[1:]-watermark_chunks[:-1]):
        #    self._weight_difference.append(np.zeros(c))

        #self._previous_batch_weight = []
        #for idx in range(len(watermark_chunks)-1):
        #    self._previous_batch_weight.append(self._watermark_values[watermark_chunks[idx]:watermark_chunks[idx+1]])

        return

    def DisableEmbedding(self, disabled):
        self._disable_embed = disabled

    def train_step(self, data):
        """
        Custom training step with watermarking algorithm between forward pass and back-propagation
        :param data: training data provided via model.fit by data generators
        :return: Dictionary with metrics and their values as in standard Keras .fit
        """

        # Condition added when moving to TF 2.5.0
        if len(data) == 2:
            x, y = data
        else:
            x, y, _ = data

        with tf.GradientTape() as tape:

            # Forward pass
            y_pred = self(x, training=True)

            # Compute the loss value (the loss function is configured in `compile()`)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights, THEN EMBED WATERMARK
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Check if watermark embedding was disabled (i.e. training a non-watermarked model)
        if not self._disable_embed:

            # Extract weights
            weights = self.weights

            def EmbedWatermark():
                """
                Embed watermark into host weights
                :return: embedded weights
                """

                # Loop over all the layers that have been chosen to embed the watermark according to their indexing
                for idx, layer_idx in enumerate(self._host_layer_indices):

                    # Select the chunk of watermark (multiplied by strength) for the current host layer
                    val_chunk = self._watermark_values[self._watermark_chunks[idx]:self._watermark_chunks[idx+1]]

                    # Select the coordinates within the current host layer
                    hosts = self._hosts[idx]
                    host_layer_type = self._host_layer_types[idx]

                    k = 0
                    for c in list(hosts):
                        for j in range(len(c)):

                            # Procedure for SeparableConv2D layers (mostly XCeptionNet)
                            if host_layer_type == tf.keras.layers.SeparableConv2D:
                                weights[layer_idx][c[j][0], c[j][1], c[j][2]].assign([val_chunk[k]])

                            # Procedure for Conv2D layers (e.g. EfficientNet, DenseNet, VGG)
                            if host_layer_type == tf.keras.layers.Conv2D:
                                weights[layer_idx][c[j][0], c[j][1], c[j][2], c[j][3]].assign([val_chunk[k]][0])

                            k += 1

                return

            # Embed watermark and replace weights
            EmbedWatermark()
            self._weights = weights

        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)

        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
