from ray.rllib import TFPolicy

# libs for testing
import gym, gym_ToricCode
import tensorflow as tf

class ToricPolicy(TFPolicy):
    """An agent policy and loss, i.e., a TFPolicy or other subclass.
    This object defines how to act in the environment, and also losses used to
    improve the policy based on its experiences. Note that both policy and
    loss are defined together for convenience, though the policy itself is
    logically separate.

    All policies can directly extend Policy, however TensorFlow users may
    find TFPolicy simpler to implement. TFPolicy also enables RLlib
    to apply TensorFlow-specific optimizations such as fusing multiple policy
    graphs and multi-GPU support.

    Attributes:
        observation_space (gym.Space): Observation space of the policy.
        action_space (gym.Space): Action space of the policy.
    """

    def __init__(self,
                observation_space,
                action_space,
                config,
                sess,
                obs_input,
                action_sampler,
                loss,
                loss_inputs,
                model=None,
                action_logp=None,
                state_inputs=None,
                state_outputs=None,
                prev_action_input=None,
                prev_reward_input=None,
                seq_lens=None,
                max_seq_len=20,
                batch_divisibility_req=1,
                update_ops=None):
        """Initialize the policy.

        Arguments:
            observation_space (gym.Space): Observation space of the env.
            action_space (gym.Space): Action space of the env.
            sess (Session): TensorFlow session to use.
            obs_input (Tensor): input placeholder for observations, of shape
                [BATCH_SIZE, obs...].
            action_sampler (Tensor): Tensor for sampling an action, of shape
                [BATCH_SIZE, action...]
            loss (Tensor): scalar policy loss output tensor.
            loss_inputs (list): a (name, placeholder) tuple for each loss
                input argument. Each placeholder name must correspond to a
                SampleBatch column key returned by postprocess_trajectory(),
                and has shape [BATCH_SIZE, data...]. These keys will be read
                from postprocessed sample batches and fed into the specified
                placeholders during loss computation.
            model (rllib.models.Model): used to integrate custom losses and
                stats from user-defined RLlib models.
            action_logp (Tensor): log probability of the sampled action.
            state_inputs (list): list of RNN state input Tensors.
            state_outputs (list): list of RNN state output Tensors.
            prev_action_input (Tensor): placeholder for previous actions
            prev_reward_input (Tensor): placeholder for previous rewards
            seq_lens (Tensor): placeholder for RNN sequence lengths, of shape
                [NUM_SEQUENCES]. Note that NUM_SEQUENCES << BATCH_SIZE. See
                policy/rnn_sequencing.py for more information.
            max_seq_len (int): max sequence length for LSTM training.
            batch_divisibility_req (int): pad all agent experiences batches to
                multiples of this value. This only has an effect if not using
                a LSTM model.
            update_ops (list): override the batchnorm update ops to run when
                applying gradients. Otherwise we run all update ops found in
                the current variable scope.
        """

        super(TFPolicy, self).__init__(observation_space, action_space, config)
        self.model = model
        self._sess = sess
        self._obs_input = obs_input
        self._prev_action_input = prev_action_input
        self._prev_reward_input = prev_reward_input
        self._sampler = action_sampler
        self._is_training = self._get_is_training_placeholder()
        self._action_logp = action_logp
        self._action_prob = (tf.exp(self._action_logp)
                             if self._action_logp is not None else None)
        self._state_inputs = state_inputs or []
        self._state_outputs = state_outputs or []
        self._seq_lens = seq_lens
        self._max_seq_len = max_seq_len
        self._batch_divisibility_req = batch_divisibility_req
        self._update_ops = update_ops
        self._stats_fetches = {}
        self._loss_input_dict = None

        self.system_size = config["size"]
        self.grid_shift  = config["grid_shift"]
        self.policy_net  = config["policy_net"]

    def compute_actions(self,
                        obs_batch,
                        state_batches,
                        prev_action_batch=None,
                        prev_reward_batch=None,
                        info_batch=None,
                        episodes=None,
                        **kwargs):

        """Computes actions for the current policy.

        Args:
            obs_batch (Union[List,np.ndarray]): Batch of observations.
            state_batches (Optional[list]): List of RNN state input batches,
                if any.
            prev_action_batch (Optional[List,np.ndarray]): Batch of previous
                action values.
            prev_reward_batch (Optional[List,np.ndarray]): Batch of previous
                rewards.
            info_batch (info): Batch of info objects.
            episodes (list): MultiAgentEpisode for each obs in obs_batch.
                This provides access to all of the internal episode state,
                which may be useful for model-based or multiagent algorithms.
            kwargs: forward compatibility placeholder

        Returns:
            actions (np.ndarray): batch of output actions, with shape like
                [BATCH_SIZE, ACTION_SHAPE].
            state_outs (list): list of RNN state output batches, if any, with
                shape like [STATE_SIZE, BATCH_SIZE].
            info (dict): dictionary of extra feature batches, if any, with
                shape like {"f1": [BATCH_SIZE, ...], "f2": [BATCH_SIZE, ...]}.
        """

        # return action batch, RNN states, extra values to include in batch
        return [self.action_space.sample() for _ in obs_batch], [], {}

    def compute_single_action(self,
                              obs,
                              state=None,
                              prev_action=None,
                              prev_reward=None,
                              info=None,
                              episode=None,
                              clip_actions=False,
                              **kwargs):
        """Unbatched version of compute_actions.

        Arguments:
            obs (obj): Single observation.
            state (list): List of RNN state inputs, if any.
            prev_action (obj): Previous action value, if any.
            prev_reward (float): Previous reward, if any.
            info (dict): info object, if any
            episode (MultiAgentEpisode): this provides access to all of the
                internal episode state, which may be useful for model-based or
                multi-agent algorithms.
            clip_actions (bool): should the action be clipped
            kwargs: forward compatibility placeholder

        Returns:
            actions (obj): single action
            state_outs (list): list of RNN state outputs, if any
            info (dict): dictionary of extra features, if any
        """
        
        perspectives = self.generatePerspective(grid_shift, self.toric_size, obs)
        
        # preprocess batch of perspectives and actions 
        perspectives = [*zip(*perspectives)] # unzip perspectives to [perspectives, position]
        batch_position_actions = perspectives[0]
        batch_perspectives = convert_to_tensor(perspectives[0], tf.float32)     

        batch_perspectives = tf.transpose(batch_perspectives, perm=[0,2,3,1]) # NCHW not supported on cpu, transpose fo NHWC
        
        output = self.policy_net(batch_perspectives)

        # TODO: See if its possible to make all this tensorflow op
        # Select action greedily
        q_values_table = output.numpy()
        row, col = np.where(q_values_table == np.max(q_values_table))
        perspective = row[0]
        max_q_action = col[0] + 1
        # TODO: Pretty
        action = [batch_position_actions[perspective][0],
                  batch_position_actions[perspective][1],
                  batch_position_actions[perspective][2],
                  max_q_action]

        return np.array(action), [], {}

    
    def generatePerspective(grid_shift, toric_size, state):
        def mod(index, shift):
            index = (index + shift) % toric_size 
            return index
        perspectives = []
        vertex_matrix = state[0,:,:]
        plaquette_matrix = state[1,:,:]
        # qubit matrix 0
        for i in range(toric_size):
            for j in range(toric_size):
                if vertex_matrix[i, j] == 1 or vertex_matrix[mod(i, 1), j] == 1 or \
                plaquette_matrix[i, j] == 1 or plaquette_matrix[i, mod(j, -1)] == 1:
                    new_state = np.roll(state, grid_shift-i, axis=1)
                    new_state = np.roll(new_state, grid_shift-j, axis=2)
                    perspectives.append((new_state, (0,i,j)))
        # qubit matrix 1
        for i in range(toric_size):
            for j in range(toric_size):
                if vertex_matrix[i,j] == 1 or vertex_matrix[i, mod(j, 1)] == 1 or \
                plaquette_matrix[i,j] == 1 or plaquette_matrix[mod(i, -1), j] == 1:
                    new_state = np.roll(state, grid_shift-i, axis=1)
                    new_state = np.roll(new_state, grid_shift-j, axis=2)
                    new_state = rotate_state(new_state) # rotate perspective clock wise
                    perspectives.append((new_state, (1,i,j)))
    
        return perspectives



    def learn_on_batch(self, samples):
        """Fused compute gradients and apply gradients call.
        Either this or the combination of compute/apply grads must be
        implemented by subclasses.

        Returns:
            grad_info: dictionary of extra metadata from compute_gradients().
        Examples:
            >>> batch = ev.sample()
            >>> ev.learn_on_batch(samples)
        """

        # implement your learning code here
        return {}  # return stats

    def get_weights(self):
        """Returns model weights.

        Returns:
            weights (obj): Serializable copy or view of model weights
        """
        return {"w": self.w}

    def set_weights(self, weights):
        """Sets model weights.

        Arguments:
            weights (obj): Serializable copy or view of model weights
        """
        self.w = weights["w"]


    def export_model(self, export_dir):
        """Export Policy to local directory for serving.
        Arguments:
            export_dir (str): Local writable directory.
        """
        raise NotImplementedError


    def export_checkpoint(self, export_dir):
        """Export Policy checkpoint to local directory.
        Argument:
            export_dir (str): Local writable directory.
        """
        raise NotImplementedError





if __name__ == "__main__":

    SYSTEM_SIZE = 3
    
    env_config = {
        "size":SYSTEM_SIZE
    }

    env = gym.make('toric-code-v0', config=env_config)

    policy_config = {
        "size":SYSTEM_SIZE,
        "grid_shift": int(SYSTEM_SIZE/2),
        "policy_net": None
    }

    sess = tf.compat.v1.Session()
    obs_input = None
    action_sampler = None
    loss = None
    loss_inputs = None

    p = ToricPolicy(env.observation_space,
                    env.action_space,
                    config=policy_config,
                    sess=sess,
                    obs_input=obs_input,
                    action_sampler=action_sampler,
                    loss=loss,
                    loss_inputs= loss_inputs
                    )