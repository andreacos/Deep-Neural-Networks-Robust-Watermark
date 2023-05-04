from XNetCustomModel import CustomModel
from XNetCustomGenerators import create_batch_generators, create_training_validation_sets, augment_on_test
import os
import logging
import cv2
from tqdm import tqdm
import numpy as np
from tensorflow.keras.layers import Dense, Dropout, Flatten, BatchNormalization
from glob import glob
import utils
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping


logging.basicConfig(
    filename='HISTORYlistener.log',
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def change_input_size(model, h, w, ch=3):
    """
    Change the input size of a pre-trained model
    :param model: the model whose input size is modified
    :param h: new input height
    :param w: new input width
    :param ch: new input depth
    :return: Updated model with new input size
    """

    model._layers[0]._batch_input_shape = (None, h, w, ch)
    new_model = tf.keras.models.model_from_json(model.to_json())

    for layer, new_layer in zip(model.layers, new_model.layers):
        new_layer.set_weights(layer.get_weights())
    return new_model


def EncodeMessage(key, w_length, msg="Lorem ipsum dolor sit amet conse", msg_length=256):
    """
    Encode a string message into a binary array where each character is converted to its binary representation
    :param key: key to reproduce random generation of the watermark
    :param w_length: watermark length
    :param msg: the message that is being encoded
    :param msg_length: lenght of the binary Lorem Ipsum sequence.
    :return: The encoded message
    """

    # This should not be necessary but I don't want to mess with previously manually entered sequences of 256 bits
    if msg_length != 256:
        msg = utils.lorem_ipsum(msg_length)

    # Convert each character into its binary representation
    bits = utils.tobits(msg)

    if len(bits) < msg_length:
        bits = bits + bits[:msg_length-len(bits)]

    # Determine the spread, that is how many times each bit of the binary message must be repeated
    spread_factor = int(np.floor(w_length / len(bits)))

    # Modulate the random sequence with the spread binary message
    return np.repeat(2*np.array(bits) - 1, spread_factor), np.array(bits)


def DecodeMessage(bits):
    return utils.frombits(bits)


def target_layer_exists(hlayer, bmodel):
    """
    Determine whether the layer that is targeted for embedding the watermark exists withing the CNN model. Currently
    supported models are XCeptionNet, DenseNet, VGG and EfficientNet
    :param hlayer: the watermark host layer
    :param bmodel: the name of the model that is being used
    :return: 1 if layer exists, 0 otherwise
    """

    if bmodel == 'exception':
        return len([x for x in tf.keras.applications.Xception(weights=None).layers if x.name == hlayer])
    elif bmodel == 'densenet':
        return len([x for x in tf.keras.applications.DenseNet169(weights=None).layers if x.name == hlayer])


def get_target_layer(hlayer, bmodel):
    """
    Gets the layer that is targeted for embedding the watermark exists withing the CNN model. Currently
    supported models are XCeptionNet, DenseNet, VGG and EfficientNet
    :param hlayer: the watermark host layer
    :param bmodel: the name of the model that is being used
    :return: the watermark host layer (keras.layers)
    """
    if bmodel == 'exception':
        return [x for x in tf.keras.applications.Xception(weights=None).layers if x.name == hlayer]
    if bmodel == 'densenet':
        return [x for x in tf.keras.applications.DenseNet169(weights=None).layers if x.name == hlayer]


def supported_layers():
    """
    Lists the tensorflow layers that currently support watermarking
    :return: List of tf.keras.layers
    """
    return [tf.keras.layers.Conv2D, tf.keras.layers.SeparableConv2D, tf.keras.layers.Concatenate]


def check_layer_support(hlayer, bmodel):
    """
    Verify whether the watermark host layer is supported by the watermark algorithm
    :param hlayer: the watermark host layer
    :param bmodel: the name of the model that is being used
    :return:
    """
    supported = False
    layer_instance = None
    t_layer = get_target_layer(hlayer, bmodel)
    for t in supported_layers():
        if not supported:
            supported = isinstance(t_layer[0], t)
            if supported:
                layer_instance = t
    return supported, layer_instance


class CnnWatermarker:

    def __init__(self, key, base_model):
        self._key = key
        self._base_model = base_model
        self._watermark = None
        self._watermark_host_layers = []
        self._watermark_length = None
        self._watermark_strength = None
        self._watermark_bits_per_filter = None
        self._watermark_host_layer_types = []
        self._model = None
        self._net_input_shape = None
        self._net_weights = 'imagenet'
        self._net_classes = 2
        return

    def get_target_layers_idx(self):
        """
        Get the index of the cnn layers hosting the watermark (index according to model.weights list)
        :return: Array whose elements are the indices of the layers hosting the watermark
        """
        indices = []
        l_indices = []
        for host_layer in self._watermark_host_layers:
            s = self._model.get_layer(host_layer).get_weights()[0].shape

            indices.append([idx for idx, x in enumerate(self._model.weights) if host_layer in x.name
                            and np.array_equal(np.array(x.shape), np.array(s))][0])

            l_indices.append([idx for idx, x in enumerate(self._model.layers) if host_layer in x.name][0])

            # print(f"Host layer {host_layer}: index {l_indices[-1]} out of {len(self._model.layers)} total layers")
            print(f"Host layer {host_layer} has {s[-1]} filters with shape {s[0:-1]} for {np.prod(s)} weights")
        return indices

    def get_target_layers_weights(self):
        """
        Get the weights of the cnn layers hosting the watermark
        :return: Array whose elements are the weights of each layer hosting the watermark
        """
        host_weights = []
        for host_layer in self._watermark_host_layers:
            host_weights.append([x.weights for x in self._model.layers if x.name == host_layer])
        return host_weights

    def get_target_layers_weights_shape(self):
        """
        Get the shape of weights of the cnn layers hosting the watermark
        :return: Array whose elements are the shape of each layer hosting the watermark
        """
        return [np.array(x[0][0].shape) for x in self.get_target_layers_weights()]

    def available_hosts(self, watermark_length):
        """
        Check if the available weights in the chosen host layer are enough to host the watermark
        :param watermark_length: embedded watermark length
        :return: True if the layer hosting the watermark have enough coefficients to host it
        """
        host_shapes = self.get_target_layers_weights_shape()

        return np.sum(np.array([np.prod(x) for x in host_shapes])) >= watermark_length

    def SetCustomModel(self, model, input_shape=None, num_classes=None):
        """
         Converts a Functional model to a CustomModel for the CnnWatermarker class
        :param model: Keras sequential model
        :param input_shape: the model's input shape
        :param num_classes: number of output classes for the model
        :return:
        """

        if isinstance(model, str):
            model = tf.keras.models.load_model(model, custom_objects={"CustomModel": CustomModel})

        # If input shape is different than the one from the loaded model, then change input layer size
        if input_shape is not None:
            model = change_input_size(model, input_shape[0], input_shape[1], input_shape[2])

        # If the number of output nodes is different than the one from the loaded model, then change output size
        if num_classes is not None and model.output.shape[-1] != num_classes:

            print(f"Replacing last Dense layer with output classes={model.output.shape[-1]} "
                  f"with new output classes={num_classes}")

            model.Trainable = True
            x = model.layers[-2].output
            predictions = Dense(num_classes, activation='softmax', name='new_pred_layer')(x)
            self._model = CustomModel(inputs=model.input, outputs=predictions)
        else:
            self._model = CustomModel(inputs=model.input, outputs=model.output)
        return

    def SetModel(self, model, input_shape=None, num_classes=None):
        """
        Set a model for the CnnWatermarker class
        :param model: Keras sequential model
        :param input_shape: the model's input shape
        :param num_classes: number of output classes for the model
        :return: Nothing
        """
        if isinstance(model, str):
            model = tf.keras.models.load_model(model, custom_objects={"CustomModel": CustomModel})

        # If input shape is different than the one from the loaded model, then change input layer size
        if input_shape is not None:
            model = change_input_size(model, input_shape[0], input_shape[1], input_shape[2])

        # If the number of output nodes is different than the one from the loaded model, then change output size
        if num_classes is not None and model.output.shape[-1] != num_classes:

            print(f"Replacing last Dense layer with output classes={model.output.shape[-1]} "
                  f"with new output classes={num_classes}")
            model.Trainable = True
            x = model.layers[-2].output
            predictions = Dense(num_classes, activation='softmax', name='new_pred_layer')(x)
            self._model = CustomModel(inputs=model.input, outputs=predictions)

        else:
            self._model = model
        return

    def ExtractMultibitWatermark(self, orig_watermark, message_bits, host_layers, spread_factor=None):
        """
        Extract a multi-bit watermark from the weights of a CNN
        :param orig_watermark:
        :param message_bits: the binary message that was embedded (non blind watermarking)
        :param host_layers: name of the network layer that hosts the coefficient
        :param spread_factor: each bit of the message is mapped to a sequence of this length
        :return: correlation coefficient with watermark, number of correct bits, recovered watermark
        """

        if spread_factor is None:
            spread_factor = int(np.floor(len(orig_watermark) / len(message_bits)))

        hosts = self.ExtractWatermarkAlt(host_layers=host_layers, watermark_length=len(orig_watermark))
        recovered_watermark = hosts

        rec_message = []
        for c in range(0, len(message_bits)):
            rec_chunk = recovered_watermark[c * spread_factor:(c + 1) * spread_factor]
            # orig_chunk = orig_watermark[c * spread_factor:(c + 1) * spread_factor]
            rec_message.append(int(np.sum(rec_chunk) > 0))

        rec_message = np.array(rec_message)

        # Return correlation between embedded watermark and host signum and bit errors
        return np.corrcoef(np.sign(recovered_watermark), orig_watermark)[0][1], \
               np.sum(rec_message != message_bits), rec_message

    def ExtractWatermark(self, host_layers, watermark_length):
        """
        Extract watermarked coefficients from a CNN
        :param host_layers: array whose elements are the layers hosting the watermark
        :param watermark_length: length of the watermark
        :return: Correlation between extracted watermark and embedded watermark and both sequences in zip() format
        """

        if self._model is None:
            raise AssertionError(f"Model not found. Use SetModel() to provide a model .")

        host_layer_weights = []
        for host_layer in host_layers:
            host_layer_weights.append([x.weights for x in self._model.layers if x.name == host_layer])

        # Get host layers shapes for each layer hosting the watermark
        host_shapes = self.get_target_layers_weights_shape()

        if self._base_model == 'exception':
            wl = watermark_length / len(host_layers)
            bpf = [int(np.ceil(np.maximum(1, wl / x[2]))) for x in host_shapes]
            self._watermark_bits_per_filter = [x if x % 2 == 0 else x + 1 for x in bpf]
            #wl = watermark_length / len(host_layers)
            #self._watermark_bits_per_filter = [int(np.ceil(np.maximum(1, wl/x[2]))) for x in host_shapes]
        else:
            self._watermark_bits_per_filter = [int(np.ceil(np.maximum(1, watermark_length/np.prod(x[2:])))) for x in host_shapes]

        self._watermark_host_layer_types = [check_layer_support(x, self._base_model)[1] for x in host_layers]

        watermark_chunk_length = [len(x) for x in np.array_split(np.arange(watermark_length), len(host_layers))]

        def recover_embedded_coefficients(key, l_weights, w_len, h_shape, h_type, bits_per_filter):
            """
            Recover embedded coefficients
            :param key: key to recover the same random locations used for watermark
            :param l_weights: watermark host layer weights
            :param w_len: length of the embedded watermark
            :param h_shape: shape of the layer hosting the watermark
            :param h_type: instance of the layer hosting the watermark (e.g. Conv2D)
            :param bits_per_filter
            :return: watermark host weights, a copy of the original watermark for correlation computation
            """
            np.random.seed(key)

            wat_coefficients = []

            k, j = 0, 0
            for i in np.arange(0, int(w_len/bits_per_filter)):
            # for i in np.arange(0, int(w_len/bits_per_filter)):
                pairs = np.array(utils.generate_pairs(xvals=range(0, h_shape[0]), yvals=range(0, h_shape[1])))
                pairs = pairs[np.random.permutation(len(pairs))][:bits_per_filter]

                for p in pairs:
                    if h_type == tf.keras.layers.SeparableConv2D:
                        host = l_weights[0][0][p[0]][p[1]][i]
                        wat_coefficients.append(tf.keras.backend.eval(host[0]))

                    elif h_type == tf.keras.layers.Conv2D:
                        host = l_weights[0][0][p[0]][p[1]][k][j]
                        wat_coefficients.append(tf.keras.backend.eval(host))

                    if k == host_shape[2] - 1:
                        k = 0
                        j += 1
                    else:
                        k += 1

            # For comparison, generate the watermark that was embedded by using the key. I generate it here rather than
            # elsewhere because Numpy seed() function is tricky elsewhere
            watermark = 2 * np.random.randint(2, size=watermark_length).flatten() - 1
            return wat_coefficients, watermark

        # Extract watermark hosts and re-generate original watermark from key
        hosts = []
        watermark = []
        for c, host_shape in enumerate(host_shapes):

            l_hosts, watermark = recover_embedded_coefficients(key=self._key, l_weights=host_layer_weights[c],
                                                               h_shape=host_shape,
                                                               h_type=self._watermark_host_layer_types[c],
                                                               w_len=watermark_chunk_length[c],
                                                               bits_per_filter=self._watermark_bits_per_filter[c])
            hosts.append(l_hosts)
            watermark = watermark

        # Return correlation coefficient and both hosts and watermarks sequence
        hosts = np.array(hosts).flatten()
        correlation = np.corrcoef(np.sign(hosts), np.sign(np.array(watermark)))[0][1]
        return correlation, zip(hosts, watermark)

    def ExtractWatermarkAlt(self, host_layers, watermark_length):
        """
        Extract watermarked coefficients from a CNN
        :param host_layers: array whose elements are the layers hosting the watermark
        :param watermark_length: length of the watermark
        :return: Correlation between extracted watermark and embedded watermark and both sequences in zip() format
        """

        if self._model is None:
            raise AssertionError(f"Model not found. Use SetModel() to provide a model .")

        host_layer_weights = []
        for host_layer in host_layers:
            host_layer_weights.append([x.weights for x in self._model.layers if x.name == host_layer])

        # Get host layers shapes for each layer hosting the watermark
        host_shapes = self.get_target_layers_weights_shape()

        if self._base_model == 'exception':
            wl = watermark_length / len(host_layers)
            bpf = [int(np.ceil(np.maximum(1, wl / x[2]))) for x in host_shapes]
            self._watermark_bits_per_filter = [x if x % 2 == 0 else x + 1 for x in bpf]
        else:
            self._watermark_bits_per_filter = [int(np.ceil(np.maximum(1, watermark_length/np.prod(x[2:])))) for x in host_shapes]

        self._watermark_host_layer_types = [check_layer_support(x, self._base_model)[1] for x in host_layers]

        watermark_chunk_length = [len(x) for x in np.array_split(np.arange(watermark_length), len(host_layers))]

        def recover_embedded_coefficients(key, l_weights, w_len, h_shape, h_type, bits_per_filter):
            """
            Recover embedded coefficients
            :param key: key to recover the same random locations used for watermark
            :param l_weights: watermark host layer weights
            :param w_len: length of the embedded watermark
            :param h_shape: shape of the layer hosting the watermark
            :param h_type: instance of the layer hosting the watermark (e.g. Conv2D)
            :param bits_per_filter
            :return: watermark host weights, a copy of the original watermark for correlation computation
            """
            np.random.seed(key)
            host_locations = []

            # Get a number of random locations equal to the length of the watermark
            k, j = 0, 0
            for i in np.arange(0, int(w_len / bits_per_filter)):
                # for i in np.arange(0, w_len):
                pairs = np.array(utils.generate_pairs(xvals=range(0, h_shape[0]), yvals=range(0, h_shape[1])))
                pairs = pairs[np.random.permutation(len(pairs))][:bits_per_filter]

                # Select one coefficient per kernel from the 3rd dimension for each 4th dimension of the stack
                host_locations.append([tuple(x) + (k, j) for x in pairs])

                if k == h_shape[2] - 1:
                    k = 0
                    j += 1
                else:
                    k += 1

            wat_coefficients = []

            for i in np.arange(0, len(host_locations)):
                for p in host_locations[i]:
                    if h_type == tf.keras.layers.SeparableConv2D:
                        host = l_weights[0][0][p[0]][p[1]][p[2]]
                        wat_coefficients.append(tf.keras.backend.eval(host[0]))

                    elif h_type == tf.keras.layers.Conv2D:
                        host = l_weights[0][0][p[0]][p[1]][k][j]
                        wat_coefficients.append(tf.keras.backend.eval(host))

                    if k == host_shape[2] - 1:
                        k = 0
                        j += 1
                    else:
                        k += 1

            return wat_coefficients

        # Extract watermark hosts and re-generate original watermark from key
        hosts = []
        watermark = []
        for c, host_shape in enumerate(host_shapes):
            l_hosts = recover_embedded_coefficients(key=self._key, l_weights=host_layer_weights[c],
                                                               h_shape=host_shape,
                                                               h_type=self._watermark_host_layer_types[c],
                                                               w_len=watermark_chunk_length[c],
                                                               bits_per_filter=self._watermark_bits_per_filter[c])
            print(len(l_hosts))
            hosts.append(l_hosts)

        # Return correlation coefficient and both hosts and watermarks sequence
        hosts = np.array(hosts).flatten()
        return hosts

    def GetAllNetWeights(self):
        """
        Get all net weights into a 1-D array
        :return: all net weights
        """
        all_weights = np.array([])
        for layer in self._model.layers:
            if isinstance(layer, tf.keras.layers.Conv2D) or \
               isinstance(layer, tf.keras.layers.SeparableConv2D):
                w = tf.keras.backend.eval(tf.squeeze(layer.weights[0])).flatten()
                all_weights = np.hstack((all_weights, w))
        return all_weights, np.var(np.array(all_weights.flatten()))

    def GetWeightsVariance(self, model, host_layer=None):
        """

        :param model:
        :param host_layer:
        :return:
        """
        self.SetCustomModel(model)

        if host_layer is None:
            _, global_var = self.GetAllNetWeights()
        else:
            host_layer_weights = [x.weights for x in self._model.layers if x.name == host_layer]
            global_var = np.var(tf.keras.backend.eval(host_layer_weights[0][0]))
        return global_var

    def CreateWatermarkStrength(self, distribution, host_layers, watermark_length):
        """
        Generates watermark strength values from a chosen distribution
        :param distribution: object of class Distribution
        :param host_layers: array whose elements are the cnn layers hosting the watermark
        :param watermark_length: watermark length
        :return: Nothing
        """

        if distribution._unwat_model == 'nowatermark':
            self._watermark_strength = np.zeros(watermark_length).flatten()
            return

        # Load the original trained model without watermark
        if distribution._unwat_model is not None:
            unwat_model = tf.keras.models.load_model(distribution._unwat_model, custom_objects={"CustomModel": CustomModel})

        # Get length of the portions of the watermark that is being embedded in each host layer.
        watermark_chunk_length = [len(x) for x in np.array_split(np.arange(watermark_length), len(host_layers))]

        # For each portion, generate a strength array from the chosen distribution
        variance_info = ["NOT WATERMARKED NETWORK target layer variance:"]

        for idx, host_layer in enumerate(host_layers):

            host_layer_weights = [x.weights for x in unwat_model.layers if x.name == host_layer]
            sigma = np.var(tf.keras.backend.eval(host_layer_weights[0][0]))
            stdev = np.sqrt(sigma)  # STANDARD DEVIATION!!!!
            ksigma = distribution._sigma_mult * stdev * (1 / np.sqrt(2))   # so that var(laplacian) = var(weights)

            distribution.GenerateValues(size=watermark_chunk_length[idx], sigma=ksigma, output_png=None)

            variance_info.append(f"Variance for non watermarked '{host_layer.upper()}' is {sigma}")
            variance_info.append(f"Std deviation for non watermarked '{host_layer.upper()}' is {stdev}")

        # Set the strength as a single array (concatenate all chunks) TAKING ABS() VALUE FOR THE WATERMARK COEFFICIENTS
        self._watermark_strength = np.abs(np.array(distribution._values).flatten())

        return variance_info

    def CreateWatermark(self, watermark, watermark_length, strength_distribution, host_layers):
        """
        Generate the watermark sequence and choose its host coefficients by means of the self._key seed
        :param watermark: None, let the algorithm choose. User-defined parameter to-be-implemented in the future
        :param watermark_length: length of the watermark sequence
        :param strength_distribution: Class of the watermark strength from distributions.py
        :param host_layers: array whose elements are the layers hosting the watermark
        :return: Nothing
        """

        if self._model is None:
            raise AssertionError(f"Initialize network with CreateNetModel() before creating watermark")

        for host_layer in host_layers:
            if not target_layer_exists(host_layer, self._base_model):
                raise AssertionError(f"Host layer {host_layer} does not exist")

            if not check_layer_support(host_layer, self._base_model)[0]:
                raise AssertionError(f"Host layer type {host_layer} watermarking is not supported. "
                                     f"Available layer types: {[x.__name__ for x in supported_layers()]}")

        self._watermark_host_layers = host_layers
        self._watermark_length = watermark_length

        variance_info = self.CreateWatermarkStrength(distribution=strength_distribution,
                                                     host_layers=host_layers,
                                                     watermark_length=watermark_length)

        self._watermark_host_layer_types = [check_layer_support(x, self._base_model)[1] for x in host_layers]

        # Get host layers shapes for each layer hosting the watermark
        host_shapes = self.get_target_layers_weights_shape()

        if self._base_model == 'exception':
            wl = watermark_length / len(host_layers)
            bpf = [int(np.ceil(np.maximum(1, wl/x[2]))) for x in host_shapes]
            self._watermark_bits_per_filter = [x if x % 2 == 0 else x+1 for x in bpf]
        else:
            self._watermark_bits_per_filter = [int(np.ceil(np.maximum(1, watermark_length/np.prod(x[2:])))) for x in host_shapes]

        watermark_chunk_length = [len(x) for x in np.array_split(np.arange(watermark_length), len(host_layers))]

        def select_embedding_coefficients(key, w_len, h_shape, bits_per_filter):
            """
            Select the weights that host the watermark
            :param key: key to generate random locations used for watermark (the same key must be used @extraction
            :param w_len: watermark length
            :param h_shape:
            :param bits_per_filter
            :return: list of locations (x, y) for the to-be-watermarked weights
            """
            np.random.seed(key)
            host_locations = []

            # Get a number of random locations equal to the length of the watermark
            k, j = 0, 0

            for i in np.arange(0, int(w_len/bits_per_filter)):

                pairs = np.array(utils.generate_pairs(xvals=range(0, h_shape[0]), yvals=range(0, h_shape[1])))
                pairs = pairs[np.random.permutation(len(pairs))][:bits_per_filter]

                # Select one coefficient per kernel from the 3rd dimension for each 4th dimension of the stack
                host_locations.append([tuple(x) + (k, j) for x in pairs])

                if k == h_shape[2] - 1:
                    k = 0
                    j += 1
                else:
                    k += 1

            return host_locations

        watermark_hosts = []
        for c, host_shape in enumerate(host_shapes):

            # Kernels with shape 1x1 on the first two dimensions (like in EfficientNet)  can't host more than one
            # watermark coefficient per kernel. Throw an error if more than one coefficient per kernel is required
            if host_shape[0] == 1 and host_shape[1] == 1 and np.prod(host_shape[2:]) < watermark_chunk_length[c]:
                raise AssertionError(f"Layer with shape [1, 1, M, N] has at most NxM hosts but watermark length is "
                                     f" {watermark_chunk_length[c]}")

            watermark_hosts.append(select_embedding_coefficients(key=self._key,
                                                                 w_len=watermark_chunk_length[c],
                                                                 h_shape=host_shape,
                                                                 bits_per_filter=self._watermark_bits_per_filter[c]))

        self._watermark = watermark

        # Setup CustomModel for embedding by providing watermarking information
        self._model.SetParams(hosts=watermark_hosts,
                              host_layer_indices=self.get_target_layers_idx(),
                              host_layer_types=self._watermark_host_layer_types,
                              watermark=watermark,
                              watermark_chunks=np.cumsum(np.hstack(([0], watermark_chunk_length))),
                              strength=self._watermark_strength)
        return variance_info

    def IsWatermarkDisabled(self):
        """
        Checks if watermark embedding is enabled
        :return: True if watermark embedding is disabled, False otherwise
        """
        return self._model.GetDisableStatus()

    def DisableEmbedding(self, disabled=True):
        """
        Disable watermark embedding (this is the switch to train a model without watermark)
        :param disabled: True to disable embedding, False otherwise
        :return: Nothing
        """
        self._model.DisableEmbedding(disabled)
        return

    def CreateNetwork(self, input_shape, weights, classes, activation=None, host_layers=None):
        """
        Initialize XCeption Net
        :param input_shape: network input shape
        :param weights: weights for XCeption, either 'imagenet' or None
        :param classes: number of output classes in the last layer
        :param activation: the activation type for all layers of the network
        :param host_layers: name of the layer that will host the watermark
        :return: Nothing
        """

        assert self._base_model in ['vgg', 'exception', 'densenet', 'efficientnet'], \
            "Base network model should be either 'xception', 'densenet', 'vgg' or 'efficientnet'."

        self._net_input_shape = input_shape
        self._net_weights = weights
        self._net_classes = classes

        if self._base_model == 'exception':
            self.Xception(input_shape=self._net_input_shape, weights=self._net_weights, num_classes=self._net_classes)
        elif self._base_model == 'densenet':
            self.Densenet(input_shape=self._net_input_shape, weights=self._net_weights, num_classes=self._net_classes)
        elif self._base_model == 'vgg':
            self.VGG(input_shape=self._net_input_shape, weights=self._net_weights, num_classes=self._net_classes)
        elif self._base_model == 'efficientnet':
            self.EfficientNet(input_shape=self._net_input_shape, weights=self._net_weights, num_classes=self._net_classes)

        return

    def Xception(self, input_shape, weights, num_classes):
        """
        Turn a pre-trained XCeption net model into a model with user-defined output classes
        :param input_shape: network input shape
        :param weights: pre-trained weights: either 'imagenet' or None
        :param num_classes: number of output classes
        :return: Keras Model
        """
        base_model = tf.keras.applications.Xception(weights=weights, include_top=False, input_shape=input_shape,
                                                    pooling='avg')
        base_model.trainable = True

        x = base_model.output
        x = Dense(256, activation='relu')(x)
        x = Dropout(0.5)(x)
        predictions = Dense(num_classes, activation='softmax')(x)

        self._model = CustomModel(inputs=base_model.input, outputs=predictions)
        return

    def Densenet(self, input_shape, weights, num_classes):
        """
        Turn a pre-trained DenseNet model into a model with user-defined output classes
        :param input_shape: network input shape
        :param weights: pre-trained weights: either 'imagenet' or None
        :param num_classes: number of output classes
        :return: Keras Model
        """
        base_model = tf.keras.applications.DenseNet169(weights=weights, include_top=False, input_shape=input_shape, pooling='avg')
        base_model.trainable = True

        x = base_model.output
        x = Flatten()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(units=512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        predictions = Dense(units=num_classes, activation='softmax')(x)

        self._model = CustomModel(inputs=base_model.input, outputs=predictions)

        return

    def CompileModel(self, learning_rate=1e-2, run_eagerly=False):
        """
        Compile model
        :return: Nothing
        """

        if self._base_model == 'vgg':
            learning_rate = 1e-5

        self._model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                            metrics=['accuracy'], run_eagerly=run_eagerly)

        return

    def Train(self, class0_dir, class1_dir, labels, batch_size, epochs, augmentation, model_dir, run_eagerly=False):
        """
        :param class0_dir: directory of the images belonging to class 0
        :param class1_dir: directory of the images belonging to class 1
        :param labels: class labels
        :param batch_size: training and validation batch size
        :param epochs: training epochs
        :param augmentation: True/False flag to enable/disable augmentation
        :param model_dir: directory where models are saved by CheckpointSaver callback
        :return: Nothing
        """

        if len(labels) > 2:
            raise AssertionError(f"This method currently supports only 2-class problems")

        # Create training and validation sets and labels
        x_train, x_validation, y_train, y_validation = create_training_validation_sets(class0_dir, class1_dir, labels)

        # Initialize batch generators
        training_generator, validation_generator = create_batch_generators(x_train=x_train, x_validation=x_validation,
                                                                           y_train=y_train, y_validation=y_validation,
                                                                           augmentation=augmentation)

        # Create model directory
        os.makedirs(os.path.join(model_dir), exist_ok=True)

        # Define callbacks
        checkpoint_saver = ModelCheckpoint(filepath=os.path.join(model_dir, 'ckpt.epoch{epoch:02d}-loss{val_loss:.2f}.h5'),
                                           monitor='val_loss', verbose=1, save_weights_only=False, save_best_only=False)

        lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=np.sqrt(0.1), cooldown=0, patience=2, min_lr=0.5e-6)

        callbacks = [lr_reducer, checkpoint_saver]

        # Model training
        self._model.fit(training_generator,
                        steps_per_epoch=int(len(x_train) // batch_size),
                        validation_data=validation_generator,
                        validation_steps=int(len(x_validation) // batch_size),
                        callbacks=callbacks,
                        epochs=epochs)

        return

    def Test(self, model_path, class0_dir, class1_dir, labels, augmentation=False, output_dir='./', log_file='log.txt'):
        """
        Test a previously trained model for a two-classes problem
        :param model_path: file path of the to-be-tested model. If None, use self._model
        :param class0_dir: directory of the images belonging to class 0
        :param class1_dir: directory of the images belonging to class 1
        :param labels: class labels
        :param augmentation: True if test must be carried out with image augmentation
        :param output_dir:
        :param log_file:
        :return: Test accuracy and prediction results for each image in zip() format
        """

        os.makedirs(output_dir, exist_ok=True)

        if len(labels) > 2:
            raise AssertionError(f"This method currently supports only 2-class problems")

        if model_path is not None:
            logging.info(f'Testing model {model_path}')
            model = tf.keras.models.load_model(model_path)
        else:
            logging.info('Testing self._model')
            model = self._model
        input_shape = model.input_shape[1:]

        # Image and labels
        class0_images = glob(os.path.join(class0_dir, '*.*'))
        class0_images = [x for x in class0_images if "print" not in x and "nvidia" not in x]
        class1_images = glob(os.path.join(class1_dir, '*.*'))
        class1_images = [x for x in class1_images if "print" not in x and "nvidia" not in x]

        class0_labels = [labels[0] for x in range(len(class0_images))]
        class1_labels = [labels[1] for x in range(len(class1_images))]

        test_images = class0_images + class1_images
        test_labels = class0_labels + class1_labels

        print(f"Found {len(test_images)} images to test ({len(class0_images)} for class 0 and {len(class1_images)} for class 1)")

        predictions = []
        pred_labels = []

        # Test
        for im_file in tqdm(test_images):

            img = cv2.imread(im_file)
            if img.shape != input_shape:
                img = cv2.resize(img, dsize=(input_shape[0], input_shape[1]))

            if augmentation:
                img = augment_on_test(img)

            # Classify
            score = model.predict(np.expand_dims(img / 255., 0))
            predictions.append(score)
            pred_labels.append(np.argmax(score, 1))

        pred_labels = np.array(pred_labels).flatten()
        test_labels = np.array(test_labels).flatten()
        lmin = min(len(pred_labels), len(test_labels))
        pred_labels = pred_labels[:lmin]
        test_labels = test_labels[:lmin]

        accuracy = np.sum(pred_labels == test_labels) / len(pred_labels)

        return accuracy
