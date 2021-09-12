from __future__ import print_function
from __future__ import absolute_import
import os
import sys
import tensorflow as tf
from ultra.ranking_model import BaseRankingModel
from ultra.ranking_model import ActivationFunctions
import ultra.utils


class MTLBiasTowerDNN(BaseRankingModel):
    """The deep neural network model for learning to rank.

    This class implements a deep neural network (DNN) and bias tower based ranking model. It's essientially a multi-layer perceptron network.

    """

    def __init__(self, hparams_str=None, **kwargs):
        """Create the network.

        Args:
            hparams_str: (String) The hyper-parameters used to build the network.
        """

        self.hparams = ultra.utils.hparams.HParams(
            # Number of neurons in each layer of a ranking_model.
            hidden_layer_sizes=[512, 256, 128],
            # Type for activation function, which could be elu, relu, sigmoid,
            # or tanh
            activation_func='elu',
            initializer='glorot',                         # Set parameter initializer
            norm="layer",                                # Set the default normalization
            position_embsize=24,                         # position embedding size
            bias_hidden_layer_sizes=[128, 64],           # hidden layers for bias
            enable_biases=[True, False],                 # enable biases of two tasks
            combine_modes=["sum", "sum"],                # combination mode of bias tower and main tower
            output_acts=["identity", "indentity"],        # activation function of output
            tasks=["click", "watchtime"],
            mtl_model="DNN",
        )
        self.hparams.parse(hparams_str)
        print("in bias tower dnn.. hparams:", self.hparams)
        self.initializer = None
        self.act_func = None
        self.layer_norm = None
        self.layer_norm_biases = [None, None]

        if self.hparams.activation_func in BaseRankingModel.ACT_FUNC_DIC:
            self.act_func = BaseRankingModel.ACT_FUNC_DIC[self.hparams.activation_func]

        if self.hparams.initializer in BaseRankingModel.INITIALIZER_DIC:
            self.initializer = BaseRankingModel.INITIALIZER_DIC[self.hparams.initializer]

        self.model_parameters = {}

    def build(self, input_list, noisy_params=None,
              noise_rate=0.05, is_training=False, **kwargs):
        """ Create the DNN model

        Args:
            input_list: (list<tf.tensor>) A list of tensors containing the features
                        for a list of documents.
            noisy_params: (dict<parameter_name, tf.variable>) A dictionary of noisy parameters to add.
            noise_rate: (float) A value specify how much noise to add.
            is_training: (bool) A flag indicating whether the model is running in training mode.

        Returns:
            A list of tf.Tensor containing the ranking scores for each instance in input_list.
        """

        with tf.variable_scope(tf.get_variable_scope(), initializer=self.initializer,
                               reuse=tf.AUTO_REUSE):
            input_data = tf.concat(input_list, axis=0)
            print("input_data shape:", input_data.get_shape().as_list())
            output_data = input_data
            output_sizes = self.hparams.hidden_layer_sizes + [1]
            output_sizes_bias = self.hparams.bias_hidden_layer_sizes + [1]

            if self.layer_norm is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
                self.layer_norm = []
                for j in range(len(output_sizes)):
                    self.layer_norm.append(BaseRankingModel.NORM_FUNC_DIC[self.hparams.norm](
                        name="layer_norm_%d" % j))
            for idx, task in enumerate(self.hparams.tasks):
                if self.hparams.enable_biases[idx] and self.layer_norm_biases[idx] is None and self.hparams.norm in BaseRankingModel.NORM_FUNC_DIC:
                    self.layer_norm_biases[idx] = []
                    for j in range(len(output_sizes_bias)):
                        self.layer_norm_biases[idx].append(BaseRankingModel.NORM_FUNC_DIC[self.hparams.norm](
                            name="layer_norm_bias_%s_%d" % (task, j)))
                output_datas = [output_data, output_data]
            for task_index, task_name in enumerate(self.hparams.tasks):
                output_data = output_datas[task_index]
                current_size = output_data.get_shape()[-1].value
                for j in range(len(output_sizes)):
                    if self.layer_norm is not None:
                        if self.hparams.norm == "layer":
                            output_data = self.layer_norm[j](
                                output_data)
                        else:
                            output_data = self.layer_norm[j](
                                output_data, training=is_training)
                    expand_W = self.get_variable(
                        "dnn_W_%d_%s" % (j, task_name), [current_size, output_sizes[j]], noisy_params=noisy_params, noise_rate=noise_rate)
                    expand_b = self.get_variable("dnn_b_%d_%s" % (j, task_name), [
                        output_sizes[j]], noisy_params=noisy_params, noise_rate=noise_rate)
                    output_data = tf.nn.bias_add(
                        tf.matmul(output_data, expand_W), expand_b)
                    # Add activation if it is a hidden layer
                    if j != len(output_sizes) - 1:
                        output_data = self.act_func(output_data)
                    current_size = output_sizes[j]
                output_datas[task_index] = output_data
        
            rank_list_size = kwargs["rank_list_size"]
            forward_only = kwargs["forward_only"]
            print("in bias tower.. forward_only:", forward_only)
            print("in bias tower.. rank_list_size:", rank_list_size)
            if not forward_only:
                # bias tower

                for task_index, task_name in enumerate(self.hparams.tasks):
                    if not self.hparams.enable_biases[task_index]:
                        continue
                    bias_inputs = []
                    input_list_size = len(input_list)
                    # position embedding
                    pos_embeddings = self.get_variable("pos_embedding_%s" %task_name, [rank_list_size, self.hparams.position_embsize])
                    for i in range(input_list_size):
                        ps_input = tf.ones(shape=[tf.shape(output_datas[0])[0] / input_list_size], dtype=tf.int32) * i  # [batch_size]
                        pos_embedding = tf.nn.embedding_lookup(pos_embeddings, ps_input) # [batch_size, position_embsize]
                        bias_inputs.append(pos_embedding)
                        print("ps_input in ", i, " of task ", task_name, " :", ps_input)
                        print("pos_embedding in ", i, "of task ", task_name, " :", pos_embedding)
                    bias_input = tf.concat(bias_inputs, axis=0)
                    print("in bias tower.. bias_input shape:", bias_input.get_shape().as_list())
           
                    output_data_bias = bias_input
                    current_size_bias = output_data_bias.get_shape()[-1].value
                    for j in range(len(output_sizes_bias)):
                        if self.layer_norm_biases[task_index] is not None:
                            if self.hparams.norm == "layer":
                                output_data_bias = self.layer_norm_biases[task_index][j](
                                    output_data_bias)
                            else:
                                output_data_bias = self.layer_norm_biases[task_index][j](
                                    output_data_bias, training=is_training)
                        expand_W = self.get_variable(
                            "bias_dnn_W_%s_%d" % (task_name, j), [current_size_bias, output_sizes_bias[j]], noisy_params=noisy_params, noise_rate=noise_rate)
                        expand_b = self.get_variable("bias_dnn_b_%s_%d" % (task_name, j), [
                            output_sizes_bias[j]], noisy_params=noisy_params, noise_rate=noise_rate)
                        output_data_bias = tf.nn.bias_add(
                            tf.matmul(output_data_bias, expand_W), expand_b)
                        # Add activation if it is a hidden layer
                        if j != len(output_sizes_bias) - 1:
                           output_data_bias = self.act_func(output_data_bias)
                        current_size_bias = output_sizes_bias[j]
                    print("output_data_bias shape of %s:%s" %(task_name, output_data_bias.get_shape().as_list()))
                    if self.hparams.combine_modes[task_index] == "sum":
                       output_datas[task_index] = tf.math.add(output_datas[task_index], output_data_bias)
                    elif self.hparams.combine_modes[task_index] == "dot":
                        output_data_bias = tf.nn.sigmoid(output_data_bias)
                        if self.hparams.output_acts[task_index] == "identity":
                            output_datas[task_index] = tf.math.multiply(output_datas[task_index], output_data_bias)
                        elif self.hparams.output_acts[task_index] == "sigmoid":
                            output_datas[task_index] = tf.nn.sigmoid(output_datas[task_index])
                            output_datas[task_index] = tf.math.multiply(output_datas[task_index], output_data_bias)
                            output_datas[task_index] = self.reverse_sigmoid(output_datas[task_index])
            output_datas = [tf.split(output_data, len(input_list), axis=0) for output_data in output_datas]
            return output_datas
