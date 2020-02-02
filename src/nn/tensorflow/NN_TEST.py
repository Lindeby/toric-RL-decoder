from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense
from ray.rllib.models.tf.tf_modelv2 import TFModelV2
import tensorflow as tf

class NN_11(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(NN_11, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        model = Sequential()
        model.add(Conv2D(input_shape=(obs_space.shape[1], obs_space.shape[1], 2), data_format='channels_last', filters=128 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=128 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=120 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=111 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=104 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=103 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=90  ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=80  ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=73  ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=71  ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=64  ,kernel_size=3, strides=1, padding='valid',use_bias=True))
        model.add(Dense(num_outputs))
        self.base_model = model
        self.register_variables(self.base_model.variables)
    
    # @tf.function
    def forward(self, input_dict, state, seq_lens):
        """Call the model with the given input tensors and state.
        Any complex observations (dicts, tuples, etc.) will be unpacked by
        __call__ before being passed to forward(). To access the flattened
        observation tensor, refer to input_dict["obs_flat"].

        This method can be called any number of times. In eager execution,
        each call to forward() will eagerly evaluate the model. In symbolic
        execution, each call to forward creates a computation graph that
        operates over the variables of this model (i.e., shares weights).
        Custom models should override this instead of __call__.

        Arguments:
            input_dict (dict): dictionary of input tensors, including "obs",
                "obs_flat", "prev_action", "prev_reward", "is_training"
            state (list): list of state tensors with sizes matching those
                returned by get_initial_state + the batch dimension
            seq_lens (Tensor): 1d tensor holding input sequence lengths

        Returns:
            (outputs, state): The model output tensor of size
                [BATCH, num_outputs]
        Sample implementation for the ``MyModelClass`` example::
            def forward(self, input_dict, state, seq_lens):
                model_out, self._value_out = self.base_model(input_dict["obs"])
                return model_out, state
        """
        s = tf.transpose(input_dict['obs'], perm=[0,2,3,1]) # NCHW not supported on cpu, transpose fo NHWC

        model_out, self._value_out = self.base_model(s)
        # model_out, self._value_out = self.base_model(state)
        
        return model_out, state

    def value_function(self):
        """Return the value function estimate for the most recent forward pass.
        Returns:
            value estimate tensor of shape [BATCH].
        """
        return self._value_out




class NN_17(TFModelV2):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        super(NN_17, self).__init__(obs_space, action_space, num_outputs, model_config, name)
        model = Sequential()
        model.add(Conv2D(input_shape=(obs_space[1], obs_space[1], 2), data_format='channels_last', filters=256 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=256 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=251 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=250 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=240 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=240 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=235 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=233 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=233 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=229 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=225 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=223 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=220 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=220 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=220 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=215 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=214 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=205 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=204 ,kernel_size=3, strides=1, padding='same', use_bias=True))
        model.add(Conv2D(filters=200 ,kernel_size=3, strides=1, padding='valid',use_bias=True))
        model.add(Dense(3))
        self.base_model = model
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out, self._value_out = self.base_model(input_dict['obs'])
        return model_out, state

    def value_function(self):
        return self._value_out